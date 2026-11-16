"""
test_trauma.py

Description: Pins the trauma rule -- its scoping, its three states, its
categories, and the counts it reports over this cohort.

Every fixture here is synthetic. No text comes from a patient report.
"""

# Standard libraries

# Non-standard libraries
import pytest

# Custom libraries
from src.core.trauma import (
    TRAUMA_NO,
    TRAUMA_PATTERNS,
    TRAUMA_UNKNOWN,
    TRAUMA_YES,
    indication_text,
    trauma_categories,
    trauma_columns,
)


def flag(text):
    """The flag alone, for the cases that do not care about the categories."""
    return trauma_categories(text)[0]


def test_a_mechanism_in_the_indication_is_trauma():
    assert flag("Indication for exam: Motor vehicle collision. Liver: Normal.") == TRAUMA_YES


def test_a_mechanism_in_the_findings_is_not():
    """The scoping is what makes the rule mean anything.

    `penetrating` is in 13,396 reports of this cohort because a penetrating
    atherosclerotic ulcer is a standard aorta phrase, and `trauma` in 15,418
    because the findings discuss it. Inside the indication they fall to 62
    and 2,640.
    """
    report = (
        "Indication: Abdominal pain. "
        "Vasculature: Penetrating atherosclerotic ulcer of the aorta. "
        "Bones: No acute traumatic injury."
    )
    assert flag(report) == TRAUMA_NO


def test_an_indication_with_no_mechanism_is_no():
    assert flag("Indication for exam: Abdominal pain. Liver: Normal.") == TRAUMA_NO


def test_a_report_with_no_indication_is_unknown():
    """36 of the 12,681 reports dictate no indication section.

    Writing `no` would put them in the medical contrast class on no evidence,
    so the flag has three states the way `template` does.
    """
    assert flag("Liver: Normal. Spleen: Normal.") == TRAUMA_UNKNOWN
    assert flag("") == TRAUMA_UNKNOWN
    assert flag(None) == TRAUMA_UNKNOWN


def test_the_additional_history_section_counts():
    """5,762 reports dictate it, and 185 of the flagged rest on it alone.

    An acute traumatic finding appears in 30.8% of those 185 against a
    16.6% cohort base rate, so they name the presenting injury.
    """
    report = "Indication: Hypotension. Additional history: GLF at home. Liver: Normal."
    assert flag(report) == TRAUMA_YES


def test_a_comorbidity_history_does_not_demote_the_trauma():
    """No clause demotes a remote history, deliberately.

    In these indications `history of` attaches to a comorbidity while the
    trauma in the same sentence is the acute event, and `status post` names
    the index event. History phrasing appears in only 105 flagged reports, and
    reading them showed most were acute.
    """
    assert flag("Indication: Motor vehicle collision. History of asthma.") == TRAUMA_YES
    assert flag("Indication: Status post MVC with chest pain.") == TRAUMA_YES


def test_acute_kidney_injury_is_not_an_injury():
    """The one measured collision with `injury`, in 3 reports of the cohort."""
    assert flag("Indication: Acute kidney injury, rising creatinine.") == TRAUMA_NO
    assert flag("Indication: AKI.") == TRAUMA_NO


def test_the_local_abbreviations_fire():
    """Three abbreviations account for eight of ten misses in a read sample.

    `PVA` is pedestrian versus auto, `GLF` a ground-level fall, and `WBCT`
    the whole-body CT a trauma activation orders.
    """
    assert flag("Indication for exam: PVA.") == TRAUMA_YES
    assert flag("Indication for exam: glf.") == TRAUMA_YES
    assert flag("Indication for exam: WBCT ordered.") == TRAUMA_YES


def test_the_category_names_what_matched():
    assert trauma_categories("Indication: MVC.")[1] == ("vehicle",)
    assert trauma_categories("Indication: Found down.")[1] == ("nonspecific",)
    assert trauma_categories("Indication: GSW to the flank.")[1] == ("penetrating",)
    assert trauma_categories("Indication: Fall from a bicycle.")[1] == ("fall", "vehicle")


def test_the_weak_category_is_separable():
    """`injury` and `found down` measure 23.1% and 22.5% acute prevalence.

    The established mechanism terms measure 25.6% and the cohort base rate is
    16.6%, so they belong in the flag and belong named apart from it.
    """
    flag_value, terms = trauma_categories("Indication: Found down, altered.")
    assert flag_value == TRAUMA_YES
    assert terms == ("nonspecific",)


@pytest.mark.parametrize("category", sorted(TRAUMA_PATTERNS))
def test_every_category_fires_on_its_own(category):
    """A category nothing can reach would read as a term list nobody uses."""
    samples = {
        "vehicle": "Indication: Motorcycle collision.",
        "fall": "Indication: Fell at home.",
        "assault": "Indication: Assaulted last night.",
        "penetrating": "Indication: Stab wound to the abdomen.",
        "named": "Indication: Blunt abdominal trauma.",
        "pathway": "Indication: WBCT ordered.",
        "nonspecific": "Indication: Multiple injuries.",
    }
    assert trauma_categories(samples[category])[1] == (category,)


def test_the_columns_join_the_categories_for_one_cell():
    assert trauma_columns("Indication: Fall from a bicycle.") == ("yes", "fall|vehicle")
    assert trauma_columns("Indication: Abdominal pain.") == ("no", "")


def test_the_indication_text_leaves_the_findings_out():
    report = "Indication: MVC. Liver: Laceration of segment 6."
    body = indication_text(report)
    assert "mvc" in body.lower()
    assert "laceration" not in body.lower()
