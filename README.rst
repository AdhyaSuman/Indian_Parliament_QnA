======================================================================================================
What Does the Indian Parliament Discuss? An Exploratory Analysis of the Question Hour in the Lok Sabha
======================================================================================================
This repository contains an example version of the code used in the paper **"What Does the Indian Parliament Discuss? An Exploratory Analysis of the Question Hour in the Lok Sabha"** by Suman Adhya and Debarshi Kumar Sanyal. The code in this repository can be used to perform your own analyses of a small subset of the TCPD-IPD dataset.

Dataset description
-------------------
The `TCPD-IPD`_ is a collection of questions and answers discussed in the Lower House of the Parliament of India during the Question Hour between 1999 and 2019. 

* **Data**: https://qh.lokdhaba.ashoka.edu.in/
* **Browse/Download Data**: https://qh.lokdhaba.ashoka.edu.in/browse-data
* **About**: https://qh.lokdhaba.ashoka.edu.in/about

.. _TCPD-IPD: https://tcpd.ashoka.edu.in/question-hour/
.. _documentation: https://qh.lokdhaba.ashoka.edu.in/static/media/qh_codebook.712c9103.pdf

::

    "TPCD-IPD: TCPD Indian Parliament Dataset (Question Hour) 1.0"
    Bhogale, Saloni. Trivedi Centre for Political Data, Ashoka University, 2019.

::

    “TPCD-IPD: TCPD Indian Parliament Codebook (Question Hour) 1.0”.
    Bhogale, Saloni. Trivedi Centre for Political Data, Ashoka University, 2019.

Tutorial
---------

.. |colab1| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/AdhyaSuman/Indian_Parliament_QnA/blob/master/Notebooks/Example_Finance.ipynb
    :alt: Open In Colab

+-----------------------------------------------------------------------------------------------+----------+
| Name                                                                                          | Link     |
+===============================================================================================+==========+
| Example of preprocessing and running the LDA and, LDAseq model on the Finance ministry dataset| |colab1| |
+-----------------------------------------------------------------------------------------------+----------+

Read the paper:

1. `ACL Anthology`_

2. `ArXiv`_

If you decide to use this resource, please cite:

.. _`ACL Anthology`: https://aclanthology.org/2022.politicalnlp-1.10/
.. _`arXiv`: https://arxiv.org/abs/2304.00235


::

    @inproceedings{adhya-sanyal-2022-indian,
    title = "What Does the {I}ndian Parliament Discuss? An Exploratory Analysis of the Question Hour in the Lok Sabha",
    author = "Adhya, Suman  and
      Sanyal, Debarshi Kumar",
    booktitle = "Proceedings of the LREC 2022 workshop on Natural Language Processing for Political Sciences",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.politicalnlp-1.10",
    pages = "72--78",
    abstract = "The TCPD-IPD dataset is a collection of questions and answers discussed in the Lower House of the Parliament of India during the Question Hour between 1999 and 2019. Although it is difficult to analyze such a huge collection manually, modern text analysis tools can provide a powerful means to navigate it. In this paper, we perform an exploratory analysis of the dataset. In particular, we present insightful corpus-level statistics and perform a more detailed analysis of three subsets of the dataset. In the latter analysis, the focus is on understanding the temporal evolution of topics using a dynamic topic model. We observe that the parliamentary conversation indeed mirrors the political and socio-economic tensions of each period.",
    }
