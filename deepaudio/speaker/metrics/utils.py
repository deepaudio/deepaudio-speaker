from tqdm import tqdm
def get_all_wavs(trials):
    uris = set()
    for uri_enroll, uri_test, _ in trials:
        uris.add(uri_enroll)
        uris.add(uri_test)
    return set(uris)


def get_all_embeddings(model, wav_trials):
    embedding = dict()
    for uri in tqdm(wav_trials):
        embedding[uri] = model.make_embedding(uri)
    return embedding
