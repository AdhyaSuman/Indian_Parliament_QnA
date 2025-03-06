What Does the Indian Parliament Discuss? An Exploratory Analysis of the Question Hour in the Lok Sabha
======================================================================================================


Welcome to the official repository for the paper **"What Does the Indian Parliament Discuss? An Exploratory Analysis of the Question Hour in the Lok Sabha"** by Suman Adhya and Debarshi Kumar Sanyal.

This repository provides an example version of the code used in the paper, allowing you to perform your own analyses using a small subset of the TCPD-IPD dataset.

---

## 📌 Dataset Description
The **TCPD-IPD dataset** is a rich collection of parliamentary questions and answers from the **Question Hour** in the **Lok Sabha** (Lower House of the Indian Parliament) spanning the years **1999 to 2019**.

🔗 **Dataset Links**:
- [**Data Overview**](https://qh.lokdhaba.ashoka.edu.in/)
- [**Browse / Download Data**](https://qh.lokdhaba.ashoka.edu.in/browse-data)
- [**About the Dataset**](https://qh.lokdhaba.ashoka.edu.in/about)

📄 **Reference**:
```
"TPCD-IPD: TCPD Indian Parliament Dataset (Question Hour) 1.0"
Bhogale, Saloni. Trivedi Centre for Political Data, Ashoka University, 2019.

"TPCD-IPD: TCPD Indian Parliament Codebook (Question Hour) 1.0"
Bhogale, Saloni. Trivedi Centre for Political Data, Ashoka University, 2019.
```

---

## 🛠️ Tutorial & Code
We provide a hands-on tutorial demonstrating how to preprocess the dataset and apply **LDA** and **LDAseq** topic models to analyze discussions in the **Finance Ministry**.

📌 **Example Notebook**:
| Analysis | Link |
|----------------------|------------|
| Preprocessing & Topic Modeling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdhyaSuman/Indian_Parliament_QnA/blob/master/Notebooks/Example_Finance.ipynb) |

---

## 📖 Read the Paper
For a comprehensive understanding of our study, check out the paper here:

- 📜 [**ACL Anthology**](https://aclanthology.org/2022.politicalnlp-1.10/)
- 📜 [**ArXiv Preprint**](https://arxiv.org/abs/2304.00235)

---

## 📌 Citation
If you find this resource useful, please cite our work:

```
@inproceedings{adhya-sanyal-2022-indian,
    title = "What Does the {I}ndian Parliament Discuss? An Exploratory Analysis of the Question Hour in the Lok Sabha",
    author = "Adhya, Suman and Sanyal, Debarshi Kumar",
    booktitle = "Proceedings of the LREC 2022 workshop on Natural Language Processing for Political Sciences",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.politicalnlp-1.10",
    pages = "72--78",
    abstract = "The TCPD-IPD dataset is a collection of questions and answers discussed in the Lower House of the Parliament of India during the Question Hour between 1999 and 2019. Although it is difficult to analyze such a huge collection manually, modern text analysis tools can provide a powerful means to navigate it. In this paper, we perform an exploratory analysis of the dataset. In particular, we present insightful corpus-level statistics and perform a more detailed analysis of three subsets of the dataset. In the latter analysis, the focus is on understanding the temporal evolution of topics using a dynamic topic model. We observe that the parliamentary conversation indeed mirrors the political and socio-economic tensions of each period.",
}
```

---

🔹 **Happy Researching!** Feel free to reach out if you have any questions. 🚀

