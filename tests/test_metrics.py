from ctxlab.metrics.qa import ExactMatch, TokenF1, normalize_answer


def test_normalize_strips_articles_and_punct() -> None:
    assert normalize_answer("The Paris.") == "paris"
    assert normalize_answer("  PARIS\nextra line") == "paris"


def test_em_golden() -> None:
    em = ExactMatch()
    assert em.score("Paris", ["Paris"]) == 1.0
    assert em.score("the paris.", ["Paris"]) == 1.0
    assert em.score("Lyon", ["Paris"]) == 0.0
    assert em.score("Paris France", ["Paris"]) == 0.0


def test_f1_golden() -> None:
    f1 = TokenF1()
    assert f1.score("Paris", ["Paris"]) == 1.0
    assert f1.score("Lyon", ["Paris"]) == 0.0
    assert f1.score("", ["Paris"]) == 0.0
    assert f1.score("", [""]) == 1.0


def test_f1_partial_is_two_thirds() -> None:
    assert abs(TokenF1().score("Paris France", ["Paris"]) - 2 / 3) < 1e-9


def test_metric_maxes_over_golds() -> None:
    assert ExactMatch().score("Paris", ["Lyon", "Paris"]) == 1.0
    assert TokenF1().score("Paris", ["Lyon", "the paris"]) == 1.0
