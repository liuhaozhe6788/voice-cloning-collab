from collections import namedtuple

Utterance = namedtuple("Utterance", "name speaker_name wav spec speaker_embed emotion_embed partial_embeds synth")
Utterance.__eq__ = lambda x, y: x.name == y.name
Utterance.__hash__ = lambda x: hash(x.name)
