from tabpfn.fit_model import main

def test_train_quick():
    main(['-C', '-E', '10', '-n', '1', '-A'])