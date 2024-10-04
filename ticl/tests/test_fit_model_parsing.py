from ticl.fit_model import main


def test_fit_model_help():
    try:
        main(['--help'])
    except SystemExit as e:
        assert e.code == 0
    else:
        assert False, "Expected SystemExit"
