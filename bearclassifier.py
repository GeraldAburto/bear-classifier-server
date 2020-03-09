from singleton import Singleton
from fastai.vision import (
    load_learner,
    open_image,
    Path,
    torch,
    defaults
)

class BearClassifier(Singleton):
    path = Path('model')
    classes = {
        'teddys': 'Teddy Bear',
        'black': 'Black Bear',
        'grizzly': 'Grizzly Bear'
    }

    def predict(self, bytes):
        defaults.device = torch.device('cpu')

        learn = load_learner(self.path)
        img = open_image(bytes)
        pred_class = learn.predict(img)

        return self.classes[str(pred_class[0])]