from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

@app.get("/")
def something(
    card_no: str,
    v1: float,
    v2: float,
    v3: float,
    v4: float,
    v5: float,
    v6: float,
    v7: float,
    v8: float,
    v9: float,
    v10: float,
    v11: float,
    v12: float,
    v13: float,
    v14: float,
    v15: float,
    v16: float,
    v17: float,
    v18: float,
    v19: float,
    v20: float,
    v21: float,
    v22: float,
    v23: float,
    v24: float,
    v25: float,
    v26: float,
    v27: float,
    v28: float,
    Amount: float,
):
    card_no = card_no.replace(" ", "")
    X_input = np.array(
        [
            np.array(
                [
                    card_no,
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7,
                    v8,
                    v9,
                    v10,
                    v11,
                    v12,
                    v13,
                    v14,
                    v15,
                    v16,
                    v17,
                    v18,
                    v19,
                    v20,
                    v21,
                    v22,
                    v23,
                    v24,
                    v25,
                    v26,
                    v27,
                    v28,
                    Amount,
                ]
            )
        ]
    )
    model = pickle.load(open("./finalized_model.sav"))
    y = model.predict(X_input)
    return {"result": bool(y[0])}
