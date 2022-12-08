


class ImageNormalization(object): 

    def __call__(self, imgs): 
        mean, std = imgs.mean(axis=0), imgs.std(axis=0)
        imgs /= 255.0

        return imgs