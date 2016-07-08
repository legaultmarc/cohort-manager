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

    Alignment aln = alignScore(s_query, s_word);

    PyObject *out = PyTuple_New(3);
    PyTuple_SET_ITEM(out, 0, PyLong_FromLong(aln.score));
    PyTuple_SET_ITEM(out, 1, PyLong_FromLong(aln.left));
    PyTuple_SET_ITEM(out, 2, PyLong_FromLong(aln.right));

    return out;
}

static PyMethodDef CStringSearchMethods[] = {
    {"align_score",
     align_score,
     METH_VARARGS,
     "Matches two strings together using local sequence alignment.\n"
     ":param query: The query string (potentially containing `word`).\n"
     ":type query: str\n\n"
     ":param word: The reference str to search for. Typically this is "
     "looped over for all possible database entries.\n"
     ":type word: str\n\n"
     ":returns: The alignment score\n"
     ":rtype: float\n"
    },
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef cstringsearchmodule = {
    PyModuleDef_HEAD_INIT,
    "c_string_search",
    NULL,
    -1,
    CStringSearchMethods
};

PyMODINIT_FUNC PyInit_c_string_search(void) {
    return PyModule_Create(&cstringsearchmodule);
}
