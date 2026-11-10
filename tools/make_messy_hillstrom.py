"""Turns the clean Hillstrom email experiment into a FAKE analyst pull with a known correct answer.

WHAT THIS IS
  We have no real campaign pull yet. This script takes one clean public file and deliberately breaks
  it into the kind of messy files an analyst might hand a marketer. The data stage should be able to
  put the pieces back together.

WHAT IS REAL AND WHAT IS INVENTED
  Real (unchanged from the public file): every customer attribute, which email each customer was
  randomly assigned, and whether they visited, bought and how much they spent.
  Invented by this script: customer IDs, the file layout, ID formats, send/visit/order dates, order
  numbers, and how a buyer's total spend is divided across orders.

WHAT "THE RIGHT ANSWER" MEANS
  answer_key/analysis_table.csv is the original public file plus an ID. It is the correct TABLE.
  answer_key/expected_results.json holds the group comparisons computed from that table. Because
  emails were randomly assigned, they are valid ESTIMATES of the email's effect, with uncertainty.
  They are not the exact true effect. The test is: does the data stage rebuild the same table and so
  reach the same numbers?

WHAT PASSING DOES NOT PROVE
  We planted these messes ourselves, so passing shows the tool handles the problems we thought of. It
  says nothing about how large or strange a real pull is. Online Retail II is the check on unplanted mess.

USAGE
  curl -o /some/empty/folder/hillstrom.csv "$SOURCE_URL"     (SOURCE_URL is below)
  python tools/make_messy_hillstrom.py --input /some/empty/folder/hillstrom.csv --out local_data/hillstrom

  Check the data's license before committing any output. local_data/ is ignored by git.
  Writing the Excel file needs openpyxl (in requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep

SOURCE_URL = ("http://www.minethatdata.com/"
              "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
SOURCE_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
SEED = 2008

NO_EMAIL = "No E-Mail"
ATTRIBUTES = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
OUTCOMES = ["visit", "conversion", "spend"]

# All invented. The public file has no dates. The window matches its two-week outcome period.
EXPORT_DATE = pd.Timestamp("2008-03-02")
SEND_DATE = pd.Timestamp("2008-03-03")
OUTCOME_WINDOW_DAYS = 14

CUSTOMER_HEADER = "Customer ID"
XLSX_NAME = "customer_export.xlsx"

ABOUT = """# Fake analyst pull, built from the Hillstrom email experiment

**This is not a real analyst pull.** A clean public file was deliberately broken into messy pieces
to test whether Databoarder's data stage can put them back together.

## What's here

- `pull_full/`: what an analyst might hand over. Give these files to the data stage.
- `pull_treated_only/`: the same files, holding only people who were emailed. The data stage should
  stop and explain that there is no comparison group.
- `answer_key/`: the correct result. **Never give these files to the data stage.**

## What is real and what is invented

| Real, unchanged from the public file | Invented by the script |
|---|---|
| Customer attributes | Customer IDs and their formats |
| Which email each customer was randomly assigned | File layout |
| Whether each customer visited or bought | Send, visit and order dates |
| Each customer's total spend | Order numbers and how spend divides across orders |

## The planted problems

1. Customer attributes are in Excel, with title rows above the header and a total row at the bottom.
2. Each file writes the same customer differently: `C-00123`, `123`, `00123`, `C00123`.
3. The email list holds only people who were emailed. Being absent means they were held out.
4. Non-buyers are absent from the purchases file and non-visitors from the visits file. Being absent
   means zero here, and the tool has to ask rather than assume.
5. Purchases have one row per order, so buyers can have several rows and need rolling up.

## What "the right answer" means

`answer_key/analysis_table.csv` is the original public file plus an ID. It is the correct **table**.

`answer_key/expected_results.json` holds comparisons computed from that table. Emails were randomly
assigned, so these are valid **estimates** with uncertainty, not the exact true effect. They are
benchmarks for checking the rebuilt table, and aren't findings. No correction for looking at several
comparisons has been applied.

The test: rebuild the same table from `pull_full/`, and reach the same numbers.

## What passing does not prove

These problems were planted, so passing shows only that the tool handles the problems we thought of.
It does not show how large or how strange a real pull is. UCI's Online Retail II data is the check on
mess nobody planted, and it has no answer key.
"""


def read_source(path: Path) -> pd.DataFrame:
    """Reads the public file, refusing anything other than the exact version this script was built for."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"{path} is not the expected Hillstrom file (its sha256 is {digest}). "
                         f"Download it again from {SOURCE_URL}.")
    return pd.read_csv(path)


def answer_table(source: pd.DataFrame) -> pd.DataFrame:
    """The correct analysis table: the public file plus an ID. IDs follow the public file's row order."""
    table = source.copy()
    table.insert(0, "customer_id", np.arange(1, len(table) + 1))
    table["emailed"] = (table["segment"] != NO_EMAIL).astype(int)
    return table


def _split_cents(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    """Divides a whole number of cents into positive parts that add back up exactly."""
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int).tolist()


def _days_after_send(n: int, rng: np.random.Generator) -> pd.Series:
    days = rng.integers(0, OUTCOME_WINDOW_DAYS, n)
    return pd.Series(SEND_DATE + pd.to_timedelta(days, unit="D")).dt.strftime("%Y-%m-%d")


def _orders(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One row per order, for buyers only. Each buyer's orders add up to their real total spend."""
    rows = []
    for customer_id, spend in table.loc[table["spend"] > 0, ["customer_id", "spend"]].itertuples(index=False):
        cents = int(round(spend * 100))
        parts = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        rows += [(int(customer_id), amount) for amount in _split_cents(cents, parts, rng)]
    return pd.DataFrame(rows, columns=["customer_id", "cents"])


def _shuffled(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def build_pull(table: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """The messy pieces, as dataframes. Each file is shuffled, so row order can't be used to match them."""
    rng = np.random.default_rng(seed)

    customers = table[["customer_id"] + ATTRIBUTES].copy()
    customers.insert(0, CUSTOMER_HEADER, customers.pop("customer_id").map(lambda n: f"C-{n:05d}"))

    emailed = table[table["emailed"] == 1]
    sends = pd.DataFrame({"cust_id": emailed["customer_id"].map(str).to_numpy(),
                          "campaign": emailed["segment"].to_numpy(),
                          "send_date": SEND_DATE.strftime("%Y-%m-%d")})

    visitors = table.loc[table["visit"] == 1, "customer_id"]
    visits = pd.DataFrame({"visitor_id": visitors.map(lambda n: f"{n:05d}").to_numpy(),
                           "visit_date": _days_after_send(len(visitors), rng).to_numpy()})

    orders = _orders(table, rng)
    purchases = pd.DataFrame({"order_id": [f"O-{i:06d}" for i in range(1, len(orders) + 1)],
                              "customer_ref": orders["customer_id"].map(lambda n: f"C{n:05d}"),
                              "order_date": _days_after_send(len(orders), rng),
                              "amount": orders["cents"] / 100})

    return {"customers": _shuffled(customers, rng), "email_sends": _shuffled(sends, rng),
            "site_visits": _shuffled(visits, rng), "purchases": _shuffled(purchases, rng)}


def _digits(ids: pd.Series) -> pd.Series:
    return ids.astype(str).str.replace(r"\D", "", regex=True).astype(int)


def treated_only(pull: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The same pull, cut down to people who were emailed: the campaign-list-only case."""
    emailed = set(_digits(pull["email_sends"]["cust_id"]))
    id_columns = {"customers": CUSTOMER_HEADER, "email_sends": "cust_id",
                  "site_visits": "visitor_id", "purchases": "customer_ref"}
    return {name: pull[name][_digits(pull[name][column]).isin(emailed)].reset_index(drop=True)
            for name, column in id_columns.items()}


def _write_customers(customers: pd.DataFrame, path: Path) -> None:
    """An Excel export with title rows above the header and a total row below the data."""
    from openpyxl import Workbook

    book = Workbook(write_only=True)
    sheet = book.create_sheet("Export")
    sheet.append(["Customer attribute export"])
    sheet.append([f"Exported {EXPORT_DATE:%Y-%m-%d}"])
    sheet.append([])
    sheet.append(list(customers.columns))
    for row in customers.astype(object).itertuples(index=False):
        sheet.append(list(row))
    sheet.append([])
    sheet.append([f"Total rows: {len(customers)}"])
    book.save(path)


def write_pull(pull: dict[str, pd.DataFrame], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    _write_customers(pull["customers"], folder / XLSX_NAME)
    for name in ("email_sends", "site_visits", "purchases"):
        pull[name].to_csv(folder / f"{name}.csv", index=False)


def read_customers(path: Path) -> pd.DataFrame:
    """Finds the real header inside the export and drops the title and total rows."""
    raw = pd.read_excel(path, header=None, dtype=object)
    header_at = raw.index[raw[0] == CUSTOMER_HEADER][0]
    data = raw.iloc[header_at + 1:].copy()
    data.columns = raw.iloc[header_at].tolist()
    return data[data["recency"].notna()].reset_index(drop=True)


def rebuild(folder: Path) -> pd.DataFrame:
    """The correct way to put a pull back together. This is the result the data stage has to match."""
    customers = read_customers(folder / XLSX_NAME)
    sends = pd.read_csv(folder / "email_sends.csv", dtype=str)
    visits = pd.read_csv(folder / "site_visits.csv", dtype=str)
    purchases = pd.read_csv(folder / "purchases.csv", dtype={"customer_ref": str})

    result = customers[ATTRIBUTES].copy()
    for column in ("recency", "history", "mens", "womens", "newbie"):
        result[column] = pd.to_numeric(result[column])
    for column in ("history_segment", "zip_code", "channel"):
        result[column] = result[column].astype(str)
    result.insert(0, "customer_id", _digits(customers[CUSTOMER_HEADER]))

    campaign = dict(zip(_digits(sends["cust_id"]), sends["campaign"]))
    result["segment"] = result["customer_id"].map(campaign).fillna(NO_EMAIL)
    result["visit"] = result["customer_id"].isin(set(_digits(visits["visitor_id"]))).astype(int)

    cents = (purchases["amount"] * 100).round().astype(int).groupby(_digits(purchases["customer_ref"])).sum()
    total = result["customer_id"].map(cents).fillna(0).astype(int)
    result["conversion"] = (total > 0).astype(int)
    result["spend"] = (total / 100).astype(float)
    result["emailed"] = (result["segment"] != NO_EMAIL).astype(int)
    return result.sort_values("customer_id").reset_index(drop=True)


def _estimate(treated: pd.Series, control: pd.Series, binary: bool) -> dict:
    """Uses the same methods as the app: Newcombe range for yes/no outcomes, Welch for amounts."""
    if binary:
        low, high = confint_proportions_2indep(int(treated.sum()), len(treated), int(control.sum()), len(control),
                                               method="newcomb", compare="diff")
    else:
        test = stats.ttest_ind(treated, control, equal_var=False)
        interval = test.confidence_interval(0.95)
        low, high = interval.low, interval.high
    return {"treated_n": int(len(treated)), "control_n": int(len(control)),
            "treated_mean": float(treated.mean()), "control_mean": float(control.mean()),
            "difference": float(treated.mean() - control.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def expected_results(table: pd.DataFrame) -> dict:
    control = table[table["segment"] == NO_EMAIL]
    arms = {"Mens E-Mail": table[table["segment"] == "Mens E-Mail"],
            "Womens E-Mail": table[table["segment"] == "Womens E-Mail"],
            "Any email": table[table["emailed"] == 1]}
    return {
        "note": ("Benchmarks for checking the rebuilt table. Randomized comparisons, so valid estimates with "
                 "uncertainty, not the exact true effect. Not corrected for multiple comparisons."),
        "source_sha256": SOURCE_SHA256,
        "seed": SEED,
        "comparisons": [{"treatment": arm, "control": NO_EMAIL, "outcome": outcome,
                         **_estimate(group[outcome], control[outcome], outcome != "spend")}
                        for arm, group in arms.items() for outcome in OUTCOMES],
    }


def same_table(rebuilt: pd.DataFrame, table: pd.DataFrame) -> bool:
    columns = list(table.columns)
    return rebuilt[columns].reset_index(drop=True).equals(table[columns].reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fake analyst pull with a known answer from Hillstrom.")
    parser.add_argument("--input", type=Path, required=True, help="the downloaded public Hillstrom CSV")
    parser.add_argument("--out", type=Path, required=True, help="output folder, outside version control")
    args = parser.parse_args()

    table = answer_table(read_source(args.input))
    pull = build_pull(table)
    write_pull(pull, args.out / "pull_full")
    write_pull(treated_only(pull), args.out / "pull_treated_only")

    key = args.out / "answer_key"
    key.mkdir(parents=True, exist_ok=True)
    table.to_csv(key / "analysis_table.csv", index=False)
    (key / "expected_results.json").write_text(json.dumps(expected_results(table), indent=2) + "\n")
    (args.out / "ABOUT.md").write_text(ABOUT)

    if not same_table(rebuild(args.out / "pull_full"), table):
        raise SystemExit("The written files don't rebuild the answer table. Don't use this output.")
    print(f"Wrote a fake analyst pull to {args.out}. Rebuilding pull_full matches the answer table exactly.")
    print({name: len(frame) for name, frame in pull.items()})


if __name__ == "__main__":
    main()
