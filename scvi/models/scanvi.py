import numpy as np
import logging
import torch
from anndata import AnnData

from scvi._compat import Literal
from scvi.models._modules.vae import VAE
from scvi.models._modules.scanvae import SCANVAE
from scvi.models._base import AbstractModelClass

from scvi.inference.inference import UnsupervisedTrainer
from scvi.inference.annotation import SemiSupervisedTrainer
from scvi.inference.annotation import AnnotationPosterior

logger = logging.getLogger(__name__)


class SCANVI(AbstractModelClass):
    """Single-cell annotation using variational inference [Xu19]_

    Inspired from M1 + M2 model, as described in (https://arxiv.org/pdf/1406.5298.pdf).

    Parameters
    ----------
    adata
        AnnData object that has been registered with scvi
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)

    Examples
    --------

    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.dataset.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.models.SCVI(adata)
    >>> vae.train(n_epochs=400)
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        unlabeled_category: str,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        use_cuda: bool = True,
        **model_kwargs,
    ):
        assert (
            "scvi_data_registry" in adata.uns.keys()
        ), "Please setup your AnnData with scvi.dataset.setup_anndata(adata) first"

        self.adata = adata
        # TODO need mapping between original column key and digitized one
        self.unlabeled_category = unlabeled_category
        summary_stats = adata.uns["scvi_summary_stats"]
        self._scvi_model = VAE(
            n_input=summary_stats["n_genes"],
            n_batch=summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            reconstruction_loss=gene_likelihood,
            **model_kwargs,
        )
        self.model = SCANVAE(
            n_input=summary_stats["n_genes"],
            n_batch=summary_stats["n_batch"],
            n_labels=summary_stats["n_labels"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            reconstruction_loss=gene_likelihood,
            **model_kwargs,
        )
        self.is_trained = False
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.batch_size = 128
        self._posterior_class = AnnotationPosterior
        self._trainer_class = SemiSupervisedTrainer

    def train(
        self,
        n_epochs_unsupervised=None,
        n_epochs_semisupervised=None,
        train_size=0.9,
        test_size=None,
        lr=1e-3,
        n_iter_kl_warmup=None,
        n_epochs_kl_warmup=400,
        metric_frequency=None,
        unsupervised_trainer_kwargs={},
        semisupervised_trainer_kwargs={},
        train_kwargs={},
    ):

        if n_epochs_unsupervised is None:
            n_epochs_unsupervised = np.min(
                [round((20000 / self.adata.shape[0]) * 400), 400]
            )
        if n_epochs_semisupervised is None:
            n_epochs_semisupervised = np.min(
                [10, np.max([2, round(n_epochs_unsupervised / 3.0)])]
            )

        self._unsupervised_trainer = UnsupervisedTrainer(
            self._model_scvi,
            self.adata,
            train_size=train_size,
            test_size=test_size,
            n_iter_kl_warmup=n_iter_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            frequency=metric_frequency,
            use_cuda=self.use_cuda,
            **unsupervised_trainer_kwargs,
        )
        self.trainer.train(n_epochs=n_epochs_unsupervised, lr=lr, **train_kwargs)

        self.model.load_state_dict(self.trainer.model.state_dict(), strict=False)

        self.trainer = SemiSupervisedTrainer(self.model, self.adata)
        unlabeled_indices = None
        labeled_indices = None

        self.trainer.unlabelled_set = self.trainer.create_posterior(
            indices=unlabeled_indices
        )
        self.trainer.labelled_set = self.trainer.create_posterior(
            indices=labeled_indices
        )
        self.trainer.train(n_epochs=n_epochs_semisupervised)

        self.is_trained = True
        self.train_indices = self.trainer.train_set.indices
        self.test_indices = self.trainer.test_set.indices
        self.validation_indices = self.trainer.validation_set.indices

    def predict(self, adata=None, indices=None, soft=False):

        post = self._make_posterior(adata=adata, indices=indices)

        _, pred = post.sequential().compute_predictions(soft=soft)

        return pred
