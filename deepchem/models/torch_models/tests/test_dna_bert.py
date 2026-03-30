import os
import random
import deepchem as dc
import numpy as np
import pytest

try:
    import torch
    from deepchem.models.torch_models.dna_bert import DNABert
except ModuleNotFoundError:
    pass


@pytest.mark.hf
def test_dnabert_pretraining(dna_regression_dataset,
                             dna_multitask_regression_dataset):
    # Pretraining in MLM mode
    from deepchem.models.torch_models.dna_bert import DNABert

    tokenizer_path = 'zhihan1996/DNABERT-2-117M'
    model = DNABert(task='mlm', tokenizer_path=tokenizer_path)
    loss = model.fit(dna_regression_dataset, nb_epoch=1)

    assert loss

    # Pretraining in Multitask Regression Mode
    model = DNABert(task='mtr', tokenizer_path=tokenizer_path, n_tasks=2)
    loss = model.fit(dna_multitask_regression_dataset, nb_epoch=1)
    assert loss


@pytest.mark.hf
def test_dnabert_finetuning(dna_regression_dataset,
                            dna_multitask_regression_dataset):
    # test regression
    tokenizer_path = 'zhihan1996/DNABERT-2-117M'
    model = DNABert(task='regression', tokenizer_path=tokenizer_path)
    loss = model.fit(dna_regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(dna_regression_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_absolute_error))
    assert loss, eval_score
    prediction = model.predict(dna_regression_dataset)
    assert prediction.shape == dna_regression_dataset.y.shape

    # test multitask regression
    model = DNABert(task='mtr', tokenizer_path=tokenizer_path, n_tasks=2)
    loss = model.fit(dna_multitask_regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(dna_multitask_regression_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_absolute_error))
    assert loss, eval_score
    prediction = model.predict(dna_multitask_regression_dataset)
    assert prediction.shape == dna_multitask_regression_dataset.y.shape

    # test classification
    y = np.random.choice([0, 1], size=dna_regression_dataset.y.shape)
    dataset = dc.data.NumpyDataset(X=dna_regression_dataset.X,
                                   y=y,
                                   w=dna_regression_dataset.w,
                                   ids=dna_regression_dataset.ids)
    model = DNABert(task='classification', tokenizer_path=tokenizer_path)
    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.recall_score))
    assert eval_score, loss
    prediction = model.predict(dataset)
    # logit scores
    assert prediction.shape == (dataset.y.shape[0], 2)


@pytest.mark.hf
def test_dnabert_load_from_pretrained(tmpdir, dna_regression_dataset):
    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')
    tokenizer_path = 'zhihan1996/DNABERT-2-117M'
    pretrain_model = DNABert(task='mlm',
                             tokenizer_path=tokenizer_path,
                             model_dir=pretrain_model_dir)
    pretrain_model.save_checkpoint()

    finetune_model = DNABert(task='regression',
                             tokenizer_path=tokenizer_path,
                             model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # check weights match
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]

    assert all(matches)


@pytest.mark.hf
def test_dnabert_save_reload(tmpdir):
    tokenizer_path = 'zhihan1996/DNABERT-2-117M'
    model = DNABert(task='regression',
                    tokenizer_path=tokenizer_path,
                    model_dir=tmpdir)
    model._ensure_built()
    model.save_checkpoint()

    model_new = DNABert(task='regression',
                        tokenizer_path=tokenizer_path,
                        model_dir=tmpdir)
    model_new.restore()

    old_state = model.model.state_dict()
    new_state = model_new.model.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]

    # all keys values should match
    assert all(matches)


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.mark.hf
def test_random_weight_initialization_regression():
    tokenizer_path = 'zhihan1996/DNABERT-2-117M'

    set_seed(10)
    model1 = DNABert(task='regression',
                     tokenizer_path=tokenizer_path,
                     n_tasks=2)

    set_seed(25)
    model2 = DNABert(task='regression',
                     tokenizer_path=tokenizer_path,
                     n_tasks=2)

    model1_state_dict = model1.model.state_dict()
    model2_state_dict = model2.model.state_dict()

    model1_keys = [
        key for key in model1_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(model1_state_dict[key], model2_state_dict[key])
        for key in model1_keys
    ]
    assert not all(matches)


@pytest.mark.hf
def test_random_weight_initialization_mlm():
    tokenizer_path = 'zhihan1996/DNABERT-2-117M'

    set_seed(10)
    model1 = DNABert(task='mlm', tokenizer_path=tokenizer_path, n_tasks=2)

    set_seed(25)
    model2 = DNABert(task='mlm', tokenizer_path=tokenizer_path, n_tasks=2)

    model1_state_dict = model1.model.state_dict()
    model2_state_dict = model2.model.state_dict()

    model1_keys = [
        key for key in model1_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(model1_state_dict[key], model2_state_dict[key])
        for key in model1_keys
    ]
    assert not all(matches)


@pytest.mark.hf
def test_random_weight_initialization_mtr():
    tokenizer_path = 'zhihan1996/DNABERT-2-117M'

    set_seed(10)
    model1 = DNABert(task='mtr', tokenizer_path=tokenizer_path, n_tasks=2)

    set_seed(25)
    model2 = DNABert(task='mtr', tokenizer_path=tokenizer_path, n_tasks=2)

    model1_state_dict = model1.model.state_dict()
    model2_state_dict = model2.model.state_dict()

    model1_keys = [
        key for key in model1_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(model1_state_dict[key], model2_state_dict[key])
        for key in model1_keys
    ]
    assert not all(matches)


@pytest.mark.hf
def test_random_weight_initialization_classification():
    tokenizer_path = 'zhihan1996/DNABERT-2-117M'

    set_seed(10)
    model1 = DNABert(task='classification',
                     tokenizer_path=tokenizer_path,
                     n_tasks=2)

    set_seed(25)
    model2 = DNABert(task='classification',
                     tokenizer_path=tokenizer_path,
                     n_tasks=2)

    model1_state_dict = model1.model.state_dict()
    model2_state_dict = model2.model.state_dict()

    model1_keys = [
        key for key in model1_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(model1_state_dict[key], model2_state_dict[key])
        for key in model1_keys
    ]
    assert not all(matches)
