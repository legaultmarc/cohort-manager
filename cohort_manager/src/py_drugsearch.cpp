#include <iostream>
#include <string>

#include <Python.h>
#include "includes/query.h"


static PyObject* align_score(PyObject *self, PyObject *args) {
    const char* query;
    const char* word;

    if (!PyArg_ParseTuple(args, "ss", &query, &word))
        return NULL;

    std::string s_query = std::string(query);
    std::string s_word = std::string(word);

    double score = alignScore(s_query, s_word);

    return PyFloat_FromDouble(score);
}

static PyMethodDef CDrugSearchMethods[] = {
    {"align_score",
     align_score,
     METH_VARARGS,
     "Searches for a drug name in a query using local sequence alignment.\n"
     ":param query: The query string (potentially containing `word`).\n"
     ":type query: str\n\n"
     ":param word: The name of the drug to search for. Typically this is "
     "looped over for all possible drugs.\n"
     ":type word: str\n\n"
     ":returns: The alignment score (1 means that `word` is a substring of "
     "`query`).\n"
     ":rtype: float\n"
    },
    {NULL, NULL, 0, NULL} // Sentinel
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef cdrugsearchmodule = {
    PyModuleDef_HEAD_INIT,
    "c_drug_search",
    NULL,
    -1,
    CDrugSearchMethods
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_c_drug_search(void) {
    return PyModule_Create(&cdrugsearchmodule);
}
#else
PyMODINIT_FUNC initc_drug_search(void) {
    (void) Py_InitModule("c_drug_search", CDrugSearchMethods);
}
#endif
