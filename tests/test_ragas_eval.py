from src.eval.ragas_eval import golden_topic


def test_golden_topic_pci():
    item = {"question": "What is cardholder data under PCI DSS?"}
    assert golden_topic(item) == "pci"


def test_golden_topic_basel():
    item = {"question": "What is the Liquidity Coverage Ratio (LCR)?"}
    assert golden_topic(item) == "basel"


def test_golden_topic_explicit_override():
    item = {"question": "Generic question", "topic": "basel"}
    assert golden_topic(item) == "basel"
