

# collect embeddings from specified closed_data
def build_embeddings_streaming(
    data,
    model,
    bufer_size=10000,
    text_processing_method=None,
    text_key=None,
    text_keys=None
):
    set_embeddings = []
    current_texts = []
    for data_point in tqdm(data):
        text = combine_text_streaming(data_point, text_key=text_key, text_keys=text_keys)
        clean_text = text
        if text_processing_method != None:
            clean_text = text_processing_method(text)
        current_texts.append(clean_text)

        if len(current_texts) >= bufer_size:
            current_embeddings = model.encode(current_texts)
            set_embeddings.append(current_embeddings)
            current_texts = []
    if len(current_texts) > 0:
        current_embeddings = model.encode(current_texts)
        set_embeddings.append(current_embeddings)
    set_embeddings = np.concatenate(set_embeddings)

    return set_embeddings