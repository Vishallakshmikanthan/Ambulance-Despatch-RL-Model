class EasyConfig:
    num_ambulances = 2
    num_hospitals = 2
    max_steps = 50
    seed = 42

    def to_dict(self):
        return {
            "n_ambulances": self.num_ambulances,
            "n_hospitals": self.num_hospitals,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }


class MediumConfig:
    num_ambulances = 4
    num_hospitals = 3
    max_steps = 80
    seed = 42

    def to_dict(self):
        return {
            "n_ambulances": self.num_ambulances,
            "n_hospitals": self.num_hospitals,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }


class HardConfig:
    num_ambulances = 6
    num_hospitals = 4
    max_steps = 120
    seed = 42

    def to_dict(self):
        return {
            "n_ambulances": self.num_ambulances,
            "n_hospitals": self.num_hospitals,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }
