"""The one-screen combine flow: suggest answers to a few plain questions, then turn them into a recipe.

The answers say which table lists the people, who got the campaign, which events count, and what it
means when someone is missing from an events file. Fixed code does everything here: suggest() reads the
files and suggests answers, review() finds anything that makes the table unsafe to build, and
compile_recipe() turns the answers into the same DataRecipe steps a person could build by hand. Building,
fingerprinting and the reproduction script then work exactly as before.

Suggestions are marked as suggestions. Nothing is built until the person has confirmed every answer.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .ingest import describe, inspect_files, suggest_read_steps
from .recipe_ops import RecipeError, _clean_key, run_steps
from .schemas import (CampaignAnswer, DataRecipe, EventsTable, MissingAnswer, PeopleAnswer, TableAnswers,
                      WindowAnswer)

# Guesses until a real analyst pull shows what's normal. See docs/plans/combine-flow.md.
UNMATCHED_SHARE = 0.01     # above this share of a file's rows matching no one, matching becomes a choice
SPREAD_DAYS = 3            # send dates further apart than this get called out next to the window choice
LINK_SHARE = 0.5           # a file counts as linked when at least half its IDs are in the people table
WINDOW_DAYS = (7, 14, 30, 60, 90)

GOT_IT = "Got it"
_POSITIVE_ID = re.compile(r"cust|user|person|member|client|account|visitor|contact|subscriber|recipient|buyer|"
                          r"shopper|(^|[\s_-])id$|^id([\s_-]|$)|ref", re.I)
_NEGATIVE_ID = re.compile(r"order|invoice|transaction|txn|sku|product|item|line|session|event|visit_?id|campaign", re.I)
_CAMPAIGN = re.compile(r"send|sent|e-?mail|mail|campaign|target|treat|offer|promo|exposure|assign|variant|arm|"
                       r"cohort|audience|recipient", re.I)
_GROUP_COLUMN = re.compile(r"got|sent|treat|group|variant|arm|campaign|e-?mail|test|control|cohort|segment|"
                           r"exposed|received", re.I)
_MONEY = re.compile(r"amount|revenue|spend|spent|sales|value|total|paid|gmv", re.I)
_DATE_SHAPE = re.compile(r"^\s*(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|"
                         r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})")

_CACHE: dict = {}
_CACHE_SIZE = 4


# ------------------------------------------------------------------------------------------- reading

def read_files(paths: list[Path]) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Reads every uploaded table once, with the header and footer rows found by code. Cached between previews."""
    paths = [Path(path) for path in paths]
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    if signature not in _CACHE:
        steps = suggest_read_steps(inspect_files(paths))
        for step in steps:
            step["status"] = "confirmed"
        tables, _ = run_steps({"steps": steps, "output": steps[0]["into"]}, paths[0].parent) if steps else ({}, [])
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[signature] = (steps, tables)
    steps, tables = _CACHE[signature]
    return [dict(step) for step in steps], dict(tables)


def clean_ids(values: pd.Series, rule: str) -> pd.Series:
    """IDs cleaned exactly as the recipe's clean_key step will clean them."""
    tables = {"t": pd.DataFrame({"v": values.astype(object).to_numpy()})}
    _clean_key({"table": "t", "column": "v", "rule": rule, "into": "k"}, tables, None)
    return pd.Series(tables["t"]["k"].to_numpy(), index=values.index)


def _letters(values: pd.Series) -> set[str]:
    found = values.dropna().astype(str).str.replace(r"[^A-Za-z]", "", regex=True).str.upper()
    return set(found[found != ""])


def _dates(values: pd.Series) -> pd.Series:
    """For suggesting only. The recipe reads dates with its own stricter rules."""
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _is_date_column(values: pd.Series) -> bool:
    text = values.dropna().astype(str).head(1000)
    if text.empty or text.str.match(_DATE_SHAPE).mean() < 0.9:
        return False
    return bool(_dates(text).notna().mean() >= 0.9)


def _is_number_column(values: pd.Series) -> bool:
    text = values.dropna().astype(str)
    return not text.empty and bool(pd.to_numeric(text, errors="coerce").notna().mean() >= 0.95)


def _id_candidates(table: pd.DataFrame) -> list[str]:
    """Columns that could hold an ID: mostly filled in, not dates, not amounts with cents, not a short list of labels."""
    found = []
    for column in table.columns:
        values = table[column].dropna().astype(str)
        if values.empty or len(values) < 0.9 * len(table) or _is_date_column(values):
            continue
        if values.str.fullmatch(r"[-+]?\d*\.\d*[1-9]\d*").mean() > 0.05:
            continue
        if values.nunique() < 50 and values.nunique() < 0.05 * len(values):
            continue
        found.append(column)
    return found


def _unique_name(base: str, taken: set[str]) -> str:
    name, number = base, 2
    while name in taken:
        name, number = f"{base}_{number}", number + 1
    taken.add(name)
    return name


# ------------------------------------------------------------------------------------------ suggesting

def _find_people(tables: dict[str, pd.DataFrame]) -> Optional[dict]:
    """The table whose IDs are unique and take in the most IDs from the other files, under the better ID rule."""
    candidates = {name: _id_candidates(table) for name, table in tables.items()}
    best = None
    for rule in ("digits_only", "lowercase"):
        keys = {(name, column): clean_ids(tables[name][column], rule)
                for name, columns in candidates.items() for column in columns}
        sets = {where: set(values.dropna()) for where, values in keys.items()}
        for (people, id_column), people_keys in keys.items():
            filled = people_keys.dropna()
            if filled.empty or filled.duplicated().any():
                continue
            own_letters = _letters(tables[people][id_column])
            links = {}
            for name in tables:
                if name == people:
                    continue
                scored = []
                for column in candidates[name]:
                    found = sets[(name, column)]
                    if not found:
                        continue
                    share = len(found & sets[(people, id_column)]) / len(found)
                    if share < LINK_SHARE:
                        continue
                    score = share + 0.05 * bool(_POSITIVE_ID.search(column)) - 0.3 * bool(_NEGATIVE_ID.search(column))
                    if rule == "digits_only" and not _letters(tables[name][column]) <= own_letters:
                        score -= 0.3    # O-000839 next to C-00123: probably an order number, not a customer
                    scored.append((score, share, column))
                if scored:
                    links[name] = max(scored)
            rank = (len(links), round(sum(share for _, share, _ in links.values()), 6),
                    not _CAMPAIGN.search(people), len(filled), bool(_POSITIVE_ID.search(id_column)))
            if best is None or rank > best["rank"]:
                best = {"rank": rank, "rule": rule, "table": people, "id_column": id_column,
                        "links": {name: column for name, (_, _, column) in links.items()}}
    return best


def _date_columns(table: pd.DataFrame, skip: str) -> list[str]:
    return [column for column in table.columns if column != skip and _is_date_column(table[column])]


def _find_campaign(tables: dict[str, pd.DataFrame], found: dict) -> Optional[dict]:
    """A linked table with one row per person and one send date (or a campaign-like name)."""
    options = []
    for name, id_column in found["links"].items():
        table = tables[name]
        keys = clean_ids(table[id_column], found["rule"]).dropna()
        if keys.empty or keys.duplicated().mean() > 0.01:
            continue
        dated = _date_columns(table, id_column)
        spread = {column: _dates(table[column]).dropna().dt.normalize().nunique() for column in dated}
        date_column = min(spread, key=spread.get) if spread else None
        hint = bool(_CAMPAIGN.search(name)) or any(_CAMPAIGN.search(column) for column in table.columns)
        if (date_column and spread[date_column] <= 3) or (not date_column and hint):
            options.append((hint, -(spread.get(date_column) or 0), name, id_column, date_column))
    if not options:
        return None
    _, _, name, id_column, date_column = max(options)
    table = tables[name]
    labels = [column for column in table.columns if column not in (id_column, date_column)
              and 2 <= table[column].dropna().nunique() <= 5]
    labels.sort(key=lambda column: not _CAMPAIGN.search(column))
    return {"table": name, "id_column": id_column, "date_column": date_column,
            "versions_column": labels[0] if labels else None}


def _group_column(table: pd.DataFrame, id_column: str) -> Optional[str]:
    for column in table.columns:
        if column != id_column and _GROUP_COLUMN.search(column) and 2 <= table[column].dropna().nunique() <= 5:
            return column
    return None


def _days_in(question: str) -> Optional[int]:
    text = question.lower()
    number = re.search(r"(\d+)\s*days?", text)
    if number:
        return int(number.group(1))
    if re.search(r"\b(two|2) weeks?\b|fortnight", text):
        return 14
    if re.search(r"\bweek\b", text):
        return 7
    if re.search(r"\bmonth\b", text):
        return 30
    return None


def _window(tables: dict, campaign: Optional[dict], events: list[EventsTable], question: str) -> dict:
    """The campaign date, and the window lengths the data fully covers."""
    starts = pd.Series(dtype="datetime64[ns]")
    if campaign and campaign.get("date_column"):
        starts = _dates(tables[campaign["table"]][campaign["date_column"]]).dropna().dt.normalize()
    event_dates = pd.concat([_dates(tables[e.table][e.date_column]).dropna() for e in events if e.date_column]
                            or [pd.Series(dtype="datetime64[ns]")])
    guessed = starts.empty
    if guessed:
        starts = event_dates.dt.normalize().head(0) if event_dates.empty else pd.Series([event_dates.min().normalize()])
    if starts.empty:
        return {"start": None, "days": 1, "options": [], "guessed": True, "first": None, "last": None, "spread": 0}
    start = starts.mode().iloc[0]
    if event_dates.empty:
        return {"start": start, "days": 1, "options": [], "guessed": guessed, "first": starts.min(), "last": None,
                "spread": 0}
    last = event_dates.max().normalize()
    full = int((last - starts.max()).days) + 1
    options = sorted({days for days in WINDOW_DAYS if days <= full} | ({full} if full >= 1 else set()))
    asked = _days_in(question)
    days = asked if asked in options else (full if full >= 1 else 1)
    return {"start": start, "days": days, "options": options, "guessed": guessed, "first": starts.min(),
            "last": last, "spread": int((starts.max() - starts.min()).days)}


def _events(tables: dict, found: dict, campaign_table: Optional[str], people_columns: list[str]) -> list[EventsTable]:
    taken = set(people_columns)
    events = []
    for name, id_column in found["links"].items():
        if name == campaign_table:
            continue
        table = tables[name]
        dated = _date_columns(table, id_column)
        date_column = max(dated, key=lambda column: table[column].notna().sum()) if dated else None
        totals = {column: _unique_name(f"total_{re.sub(r'[^0-9a-zA-Z]+', '_', column).strip('_').lower()}", taken)
                  for column in table.columns
                  if column not in (id_column, date_column) and _MONEY.search(column) and _is_number_column(table[column])}
        events.append(EventsTable(table=name, id_column=id_column, date_column=date_column,
                                  any_into=_unique_name(f"any_{name}", taken), totals=totals))
    return events


def suggest(paths: list[Path], question: str = "") -> dict:
    """Suggested answers, written by code from the files themselves. Everything comes back as a suggestion."""
    _, tables = read_files(paths)
    blocked, notes = [], []
    found = _find_people(tables) if tables else None
    if not found:
        return {"answers": None, "notes": notes, "options": {},
                "blocked": ["Couldn't find a list of people with one row each, so the files can't be put together "
                            "this way. You can still build the table step by step."]}
    people_table = tables[found["table"]]
    campaign = _find_campaign(tables, found)
    if campaign:
        answer = CampaignAnswer(kind="list", table=campaign["table"], id_column=campaign["id_column"],
                                versions_column=campaign["versions_column"], date_column=campaign["date_column"])
    else:
        column = _group_column(people_table, found["id_column"])
        if not column:
            return {"answers": None, "notes": notes, "options": {},
                    "blocked": ["These files don't show who got the campaign. Ask your analyst for the list of people "
                                "who got it, or build the table step by step."],
                    "note": analyst_note(["Could you also send the list of people who got the campaign, with their "
                                          "IDs and the date each was sent it?"])}
        answer = CampaignAnswer(kind="column", versions_column=column)
    events = _events(tables, found, campaign and campaign["table"], [c for c in people_table.columns])
    window = _window(tables, campaign, events, question)
    if window["start"] is None:
        return {"answers": None, "notes": notes, "options": {},
                "blocked": ["Couldn't find any dates, so there's no way to tell when the campaign started."]}
    if window["guessed"]:
        notes.append(f"The files don't give a campaign date, so the earliest event date, "
                     f"{window['start']:%Y-%m-%d}, is suggested. Please check it.")
    answers = TableAnswers(
        people=PeopleAnswer(table=found["table"], id_column=found["id_column"]), campaign=answer,
        window=WindowAnswer(start_date=f"{window['start']:%Y-%m-%d}", days=window["days"]),
        missing=MissingAnswer(means="none"), events=events, id_rule=found["rule"])
    people_options = [{"table": name, "id_column": column} for name, column in
                       [(found["table"], found["id_column"])] + [(n, c) for n, c in found["links"].items()]
                       if not clean_ids(tables[name][column], found["rule"]).dropna().duplicated().any()]
    return {"answers": answers.model_dump(), "blocked": blocked, "notes": notes,
            "options": {"people": people_options, "window_days": window["options"],
                        "versions": sorted(map(str, tables[campaign["table"]][campaign["versions_column"]]
                                               .dropna().unique())) if campaign and campaign["versions_column"] else []}}


# ------------------------------------------------------------------------------------------ reviewing

def _check_names(answers: TableAnswers, tables: dict) -> None:
    """Every table and column an answer names must exist, so nothing (AI or person) can point at made-up data."""
    def need(table: Optional[str], column: Optional[str] = None) -> None:
        if table is None:
            return
        if table not in tables:
            raise RecipeError(f"There's no table called '{table}' in your files.")
        if column is not None and column not in tables[table].columns:
            raise RecipeError(f"The table '{table}' has no column called '{column}'.")

    need(answers.people.table, answers.people.id_column)
    campaign = answers.campaign
    if campaign.kind == "list":
        if not campaign.table or not campaign.id_column:
            raise RecipeError("Say which file lists the people who got the campaign.")
        need(campaign.table, campaign.id_column)
        need(campaign.table, campaign.versions_column)
        need(campaign.table, campaign.date_column)
    else:
        if not campaign.versions_column:
            raise RecipeError("Say which column shows who got the campaign.")
        need(answers.people.table, campaign.versions_column)
    for events in answers.events:
        need(events.table, events.id_column)
        need(events.table, events.date_column)
        for column in events.totals:
            need(events.table, column)


def review(answers: TableAnswers, tables: dict) -> dict:
    """What stops the table being built, which extra choices are needed, and what to ask the analyst."""
    _check_names(answers, tables)
    rule, blocked, asks, risks, info = answers.id_rule, [], [], [], []
    people = answers.people
    people_keys = clean_ids(tables[people.table][people.id_column], rule)
    repeats = int(people_keys.dropna().duplicated().sum())
    if repeats:
        blocked.append(f"Some people appear more than once in {people.table} ({repeats:,} extra rows), so the table "
                       "wouldn't have one row per person. Choose a different list of people, or build the table step "
                       "by step.")
    known = set(people_keys.dropna())

    linked = [(e.table, e.id_column) for e in answers.events]
    if answers.campaign.kind == "list":
        linked.insert(0, (answers.campaign.table, answers.campaign.id_column))
    if rule == "digits_only":
        letters: dict[str, set] = {}
        for table, column in [(people.table, people.id_column)] + linked:
            raw = tables[table][column]
            keys = clean_ids(raw, rule)
            marks = raw.astype(str).str.replace(r"[^A-Za-z]", "", regex=True).str.upper()
            for key, mark in zip(keys, marks):
                if isinstance(key, str) and mark:
                    letters.setdefault(key, set()).add(mark)
        clashes = [key for key, marks in letters.items() if len(marks) > 1]
        if clashes:
            blocked.append(f"{len(clashes):,} IDs would be treated as the same person even though they're written with "
                           "different letters (like A-123 and B-123). Ask your analyst whether these are the same "
                           "people, or build the table step by step.")
            asks.append("Some IDs differ only by their letters, like A-123 and B-123. Are those the same person?")

    unmatched = []
    for table, column in linked:
        raw = tables[table][column]
        keys = clean_ids(raw, rule)
        missing = int((raw.notna() & (keys.isna() | ~keys.isin(known))).sum())
        if missing:
            share = missing / max(int(raw.notna().sum()), 1)
            unmatched.append({"table": table, "rows": missing, "of": int(raw.notna().sum()), "share": share})
    risky = [item for item in unmatched if item["share"] > UNMATCHED_SHARE]
    for item in unmatched:
        text = f"{item['rows']:,} of {item['of']:,} rows in {item['table']} match no one in {people.table}."
        (risks if item in risky else info).append({"kind": "match", "text": text})
    if risky and answers.match and answers.match.action == "ask":
        names = ", ".join(item["table"] for item in risky)
        blocked.append("Waiting on your analyst about rows that match no one.")
        asks.append(f"Some rows in {names} don't match anyone in {people.table}. Are those people missing from "
                    f"{people.table}, or are their IDs written differently?")

    campaign = answers.campaign
    if campaign.kind == "list":
        got = clean_ids(tables[campaign.table][campaign.id_column], rule).dropna()
        repeats = int(got.duplicated().sum())
        if repeats:
            blocked.append(f"{repeats:,} people appear in {campaign.table} more than once, so it isn't clear which "
                           "version they got. Ask your analyst, or build the table step by step.")
            asks.append(f"Some people appear in {campaign.table} more than once. Which version did each of them get?")
        if known and known <= set(got):
            blocked.append(f"Everyone in {people.table} is also in {campaign.table}, so everyone got the campaign and "
                           "there's no one to compare with.")
            asks.append(f"Could you also send the people who were held out of the campaign? Same columns as "
                        f"{people.table}, plus their activity over the same dates. Without them I can't compare "
                        "people who got it with people who didn't.")
        if campaign.date_column:
            sent = _dates(tables[campaign.table][campaign.date_column]).dropna()
            if not sent.empty and (sent.max() - sent.min()).days > SPREAD_DAYS:
                info.append({"kind": "window", "text": f"Send dates run from {sent.min():%Y-%m-%d} to "
                             f"{sent.max():%Y-%m-%d}. People who weren't sent anything start on "
                             f"{answers.window.start_date}."})
    elif tables[people.table][campaign.versions_column].dropna().nunique() < 2:
        blocked.append(f"Everyone has the same value in '{campaign.versions_column}', so there's no one to compare with.")

    dated = [_dates(tables[e.table][e.date_column]).dropna() for e in answers.events if e.date_column]
    last = max((values.max() for values in dated if not values.empty), default=None)
    if last is not None:
        latest = pd.Timestamp(answers.window.start_date)
        if campaign.kind == "list" and campaign.date_column:
            sent = _dates(tables[campaign.table][campaign.date_column]).dropna()
            latest = max(latest, sent.max()) if not sent.empty else latest
        covered = int((last.normalize() - latest.normalize()).days) + 1
        if answers.window.days > covered:
            blocked.append(f"The activity files end on {last:%Y-%m-%d}, so a {answers.window.days}-day window isn't "
                           f"complete for everyone and would undercount. Choose {max(covered, 1)} days or fewer, or ask "
                           "your analyst for later data.")

    negative = []
    for events in answers.events:
        for column in events.totals:
            values = pd.to_numeric(tables[events.table][column], errors="coerce")
            count = int((values < 0).sum())
            if count:
                negative.append(f"{count:,} rows in {events.table} have a negative '{column}', such as refunds.")
    for text in negative:
        risks.append({"kind": "returns", "text": text})

    if answers.missing.means == "ask":
        blocked.append("Waiting on your analyst about people missing from the activity files.")
        names = " and ".join(e.table for e in answers.events) or "the activity files"
        asks.append(f"People missing from {names}: does that mean they had no activity in that period, or that their "
                    "rows weren't included?")

    needed = ["people", "campaign", "window", "missing"]
    if risky:
        needed.append("match")
    if negative:
        needed.append("returns")
    waiting = [name for name in needed if getattr(answers, name) is None or getattr(answers, name).status != "confirmed"]
    return {"blocked": blocked, "risks": risks, "info": info, "waiting": waiting, "needs": needed,
            "note": analyst_note(asks) if asks else None, "ready": not blocked and not waiting}


def analyst_note(asks: list[str]) -> str:
    """A note the person can send to their analyst. Written by fixed code, one sentence per problem."""
    numbered = asks if len(asks) == 1 else [f"{number}. {ask}" for number, ask in enumerate(asks, 1)]
    return "Hi, thanks for the files.\n\n" + "\n\n".join(numbered) + "\n\nThanks!"


# ----------------------------------------------------------------------------------------- compiling

def compile_recipe(answers: TableAnswers, read_steps: list[dict], tables: dict, confirmed: bool = False) -> DataRecipe:
    """Turns the answers into fixed recipe steps. The recipe, not the answers, is what gets run and reproduced."""
    _check_names(answers, tables)
    status = "confirmed" if confirmed else "proposed"
    people, campaign, window = answers.people, answers.campaign, answers.window
    used = [people.table] + ([campaign.table] if campaign.kind == "list" else []) + [e.table for e in answers.events]
    used = list(dict.fromkeys(used))
    taken = {column for name in used for column in tables[name].columns}
    key = _unique_name("person_id", taken)
    steps = [{**step, "status": status} for step in read_steps if step["into"] in used]

    def add(**step) -> None:
        steps.append({"status": status, **step})

    ids = {people.table: people.id_column}
    if campaign.kind == "list":
        ids[campaign.table] = campaign.id_column
    ids.update({e.table: e.id_column for e in answers.events})
    for table in used:
        add(op="clean_key", table=table, column=ids[table], rule=answers.id_rule, into=key)
    keep = [key] + [column for column in tables[people.table].columns if column != people.id_column]
    add(op="choose_columns", table=people.table, columns={column: column for column in keep})

    start_column = None
    group = campaign.versions_column
    if campaign.kind == "list":
        group = _unique_name("campaign_group", taken)
        separate = bool(campaign.versions_column) and campaign.versions == "separate"
        bring = {}
        if separate:
            bring[campaign.versions_column] = group
        if campaign.date_column:
            start_column = _unique_name("campaign_start", taken)
            bring[campaign.date_column] = start_column
        if bring:
            add(op="match", left=people.table, right=campaign.table, left_key=key, right_key=key, bring=bring)
        if separate:
            add(op="fill_missing", table=people.table, column=group, value=campaign.none_label)
        else:
            add(op="flag_membership", table=people.table, key=key, other=campaign.table, other_key=key, into=group,
                yes=GOT_IT, no=campaign.none_label)
        got = _unique_name("got_campaign", taken)
        add(op="flag_membership", table=people.table, key=key, other=campaign.table, other_key=key, into=got,
            yes="1", no="0")
        keep += [group, got]

    for events in answers.events:
        if answers.returns and answers.returns.action == "leave_out":
            for column in events.totals:
                add(op="filter", table=events.table, column=column, test="at_least", value="0", compare_as="number")
        if events.date_column:
            measures = [{"how": "any_event", "into": events.any_into}]
            measures += [{"how": "sum", "column": column, "into": into} for column, into in events.totals.items()]
            add(op="count_within_window", table=people.table, key=key, start_column=start_column,
                default_start=window.start_date, events=events.table, event_key=key, event_date=events.date_column,
                days=window.days, measures=measures, no_events_value="0", date_order=window.date_order)
        else:
            add(op="flag_membership", table=people.table, key=key, other=events.table, other_key=key,
                into=events.any_into, yes="1", no="0")
            if events.totals:
                per_person = _unique_name(f"{events.table}_per_person", set(used) | taken)
                add(op="roll_up", table=events.table, by=key, into=per_person,
                    measures=[{"how": "sum", "column": column, "into": into} for column, into in events.totals.items()])
                add(op="match", left=people.table, right=per_person, left_key=key, right_key=key,
                    bring={into: into for into in events.totals.values()})
                for into in events.totals.values():
                    add(op="fill_missing", table=people.table, column=into, value="0")
        keep += [events.any_into] + list(events.totals.values())
    if start_column:
        add(op="choose_columns", table=people.table, columns={column: column for column in keep})
    return DataRecipe(steps=steps, output=people.table)


def group_column(answers: TableAnswers, recipe: DataRecipe) -> str:
    """The column of the finished table that says who got the campaign."""
    if answers.campaign.kind == "column":
        return answers.campaign.versions_column
    for step in recipe.steps:
        if step.op in ("fill_missing", "flag_membership") and step.table == answers.people.table:
            return step.column if step.op == "fill_missing" else step.into
    raise RecipeError("Couldn't find the column saying who got the campaign.")


def decisions(answers: TableAnswers) -> list[str]:
    """The person's choices in plain words, for the report and the top of the reproduction script."""
    campaign, window = answers.campaign, answers.window
    lines = [f"Who belongs in the table: everyone in {answers.people.table}, one row each "
             f"(ID column '{answers.people.id_column}')."]
    if campaign.kind == "list":
        versions = (f", with each version in '{campaign.versions_column}' kept separate"
                    if campaign.versions_column and campaign.versions == "separate" else "")
        lines.append(f"Who got the campaign: people in {campaign.table}{versions}. Everyone else is "
                     f"'{campaign.none_label}'.")
    else:
        lines.append(f"Who got the campaign: shown by '{campaign.versions_column}' in {answers.people.table}.")
    start = ("each person's send date, or " if campaign.kind == "list" and campaign.date_column else "") + window.start_date
    lines.append(f"Which activity counts: from {start}, for {window.days} day(s), counting the first day.")
    lines.append("People missing from an activity file: counted as no activity (0)." if answers.missing.means == "none"
                 else "People missing from an activity file: not decided yet.")
    if answers.match:
        lines.append("Rows that match no one: left out." if answers.match.action == "leave_out"
                     else "Rows that match no one: not decided yet.")
    if answers.returns:
        lines.append("Negative amounts, such as refunds: " + ("they reduce the totals." if answers.returns.action == "count"
                                                              else "left out."))
    return lines


# ------------------------------------------------------------------------------------------ previewing

def _numbers(final: pd.DataFrame, answers: TableAnswers, group: str) -> list[dict]:
    people = len(final)
    numbers = [{"key": "people", "label": "People", "value": f"{people:,}", "sub": "one row each"}]
    if people == 0:
        return numbers
    counts = final[group].astype(str).value_counts()
    for label, count in counts.items():
        numbers.append({"key": f"group:{label}", "label": label, "value": f"{100 * count / people:.0f}%",
                        "sub": f"{count:,} people"})
    for events in answers.events:
        any_events = pd.to_numeric(final[events.any_into], errors="coerce")
        numbers.append({"key": events.any_into, "label": f"Any {events.table}", "value": f"{100 * any_events.mean():.1f}%",
                        "sub": f"{int(any_events.sum()):,} people"})
        for column, into in events.totals.items():
            total = pd.to_numeric(final[into], errors="coerce")
            numbers.append({"key": into, "label": f"{column} per person", "value": f"{total.mean():,.2f}",
                            "sub": "including 0 for people with none"})
    return numbers


def preview(answers: TableAnswers, paths: list[Path], sample_rows: int = 5) -> dict:
    """Builds the table from the answers, confirmed or not, to show what they'd give. Nothing is frozen."""
    read_steps, tables = read_files(paths)
    checked = review(answers, tables)
    recipe = compile_recipe(answers, read_steps, tables)
    later = [step.model_dump() for step in recipe.steps if step.op != "read_table"]
    for step in later:
        step["status"] = "confirmed"
    fixes = [{"text": describe(step), "note": ""} for step in recipe.steps if step.op == "read_table"]
    try:
        built, log = run_steps({"steps": later, "output": recipe.output}, Path(paths[0]).parent, tables)
    except RecipeError as problem:
        return {**checked, "ready": False, "error": str(problem), "numbers": [], "sample": [], "columns": [],
                "fixes": fixes, "decisions": decisions(answers)}
    for step, entry in zip(recipe.steps[len(recipe.steps) - len(later):], log):
        fixes.append({"text": describe(step), "note": entry["note"], "rows_before": entry["rows_before"],
                      "rows_after": entry["rows_after"]})
    final = built[recipe.output]
    group = group_column(answers, recipe)
    head = final.head(sample_rows)
    return {**checked, "error": None, "numbers": _numbers(final, answers, group), "group_column": group,
            "columns": list(final.columns), "sample": head.astype(object).where(head.notna(), None).to_dict("records"),
            "fixes": fixes, "decisions": decisions(answers)}
