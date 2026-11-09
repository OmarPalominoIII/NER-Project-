from gliner import GLiNER

def load_gliner_model(model_name: str = "gliner-community/gliner_small-v2.5"):
    model = GLiNER.from_pretrained(model_name)
    return model

def predict_entities(model: GLiNER, text: str, labels: list, threshold: float = 0.5):
    return model.predict_entities(text, labels, threshold)

def predict_dataset(model: GLiNER, dataset: list, labels: list, threshold: float = 0.5):
    predictions = []

    for record in dataset:
        text = record.get("text", "")
        doc_id = record.get("id", "")

        entities = model.predict_entities(text, labels, threshold = threshold)
        pred_record = {
            "id": doc_id,
            "text": text,
            "entities": entities
        }
        predictions.append(pred_record)
    return predictions