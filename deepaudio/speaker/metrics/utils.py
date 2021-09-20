def get_all_wavs(trials, wav_dict):
    uris = set()
    wav_trials = dict()
    for uri_enroll, uri_test, _ in trials:
        uris.add(uri_enroll)
        uris.add(uri_test)
    for uri in uris:
        if uri not in wav_dict:
            msg = f'{uri} is not existing'
            raise ValueError(msg)
        else:
            wav_trials[uri] = wav_dict[uri]
    return wav_trials


def get_all_embeddings(model, wav_trials):
    embedding_trials = dict()
    for uri in wav_trials:
        wav = wav_trials[uri]
        embedding_trials[uri] = model.make_embedding(wav)
    return embedding_trials
