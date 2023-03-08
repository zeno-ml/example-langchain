import pickle

import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain

from zeno import (
    DistillReturn,
    MetricReturn,
    ModelReturn,
    ZenoOptions,
    distill,
    metric,
    model,
)


@model
def get_model(model_name):
    # Blendle Notion chatbot example from:
    # https://github.com/hwchase17/chat-langchain-notion

    index = faiss.read_index("./docs.index")
    with open("./faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
    store.index = index
    chain = VectorDBQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=0), vectorstore=store
    )

    def pred(df, ops: ZenoOptions):
        res = []
        for question in df[ops.data_column]:
            result = chain({"question": question})
            res.append(
                "Answer: {}\nSources: {}".format(result["answer"], result["sources"])
            )
        return ModelReturn(model_output=res)

    return pred


@distill
def correct(df, ops: ZenoOptions):
    return DistillReturn(
        distill_output=df.apply(
            lambda x: x[ops.label_column].lower() in x[ops.output_column].lower(),
            axis=1,
        )
    )


@metric
def accuracy(df, ops: ZenoOptions):
    return MetricReturn(metric=df[ops.distill_columns["correct"]].astype(int).mean())
