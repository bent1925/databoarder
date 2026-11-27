"""Science mode in a real browser: load a sample, make every choice, confirm the read-back, and read the report.

Needs Playwright and its Chromium browser (see test_combine_screen.py). Without them these tests are skipped.
"""
import pytest

pytest.importorskip("playwright.sync_api", reason="Playwright isn't installed")


@pytest.fixture
def lab(browser, server):
    context = browser.new_context()
    opened = context.new_page()
    errors = []
    opened.on("pageerror", lambda error: errors.append(str(error)))
    opened.on("console", lambda message: message.type == "error" and not message.text.startswith(
        "Failed to load resource") and errors.append(message.text))
    opened.goto(server + "/science")
    yield opened
    context.close()
    assert errors == [], errors


def pick(page, choice, label):
    page.locator(f"#setup [data-seg={choice}]", has_text=label).first.click()


def test_one_way_anova_with_dunnett(lab):
    lab.click("text=Tumour volume by dose")
    lab.click("text=Do groups differ?")
    assert lab.locator("#setup .tag.guess").count() >= 1                 # guessed columns are marked
    assert lab.locator("#check").is_disabled()                          # nothing runs until every call is made
    pick(lab, "paired", "No, different subjects")
    pick(lab, "method", "Averages")
    pick(lab, "equal_spread", "Yes")                                    # ordinary ANOVA
    pick(lab, "followup", "Each group vs a control")
    lab.select_option("select[data-sel=control]", "Vehicle")
    pick(lab, "assignment", "Yes, at random")
    lab.click("#check")
    lab.wait_for_selector("#s4:not(.hidden)")
    text = lab.inner_text("#readback")
    assert "ordinary one-way ANOVA" in text and "Dunnett" in text and "'Vehicle'" in text
    lab.click("#run")
    lab.wait_for_selector("#report:not(.hidden) .verdict")
    assert "differ" in lab.inner_text("#report .verdict h3")
    assert lab.locator("#report svg circle").count() == 31             # 32 mice, one empty cell
    assert lab.locator("#report tr.clear").count() >= 2
    lab.click("text=♻️ Reproduce")
    assert "copied word for word" in lab.inner_text("#report pre")


def test_changing_a_choice_hides_the_old_read_back(lab):
    lab.click("text=Infections, vaccine vs placebo")
    lab.click("text=Are categories related?")
    pick(lab, "event", "Infected")
    pick(lab, "assignment", "Yes, at random")
    lab.click("#check")
    lab.wait_for_selector("#s4:not(.hidden)")
    lab.click("#swap")
    assert lab.locator("#s4").is_hidden()
    lab.click("#check")
    lab.wait_for_selector("#s4:not(.hidden)")
    assert "in 'Placebo' against 'Vaccinated'" in lab.inner_text("#readback")


def test_paste_and_problems_are_explained(lab):
    lab.click("#showpaste")
    lab.fill("#paste", "Before\tAfter\n1\t\n\t2\n3\t4\n5\t6")
    lab.click("#usepaste")
    lab.click("text=Do groups differ?")
    pick(lab, "paired", "Yes, the same subjects")
    pick(lab, "method", "Ranks")
    pick(lab, "assignment", "No")
    lab.click("#check")
    lab.wait_for_selector("#s4 .problem")
    assert "Fewer than 3 subjects" in lab.inner_text("#s4 .problem")
    assert lab.locator("#run").is_disabled()


def test_the_page_rounds_numbers_exactly_as_the_report_does(lab):
    from app.science.report import num
    values = [762.25, 1.0005, 143.0, 12345.6, 9999.5, 0.012345, 0.0001234, -44.625, 2.5e6, 806.875, 0.6895, 19.98]
    assert lab.evaluate("(xs) => xs.map(fmt)", values) == [num(x) for x in values]
