from VQA.FL_Method.fedRepRag import train_fedRepRag_stage1, train_fedRepRag_stage2


FL_METHOD_REGISTRY = {
    "fedreprag": {
        "stage1": train_fedRepRag_stage1,
        "stage2": train_fedRepRag_stage2,
    },
}