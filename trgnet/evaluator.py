from tqdm import tqdm


class Evaluator:
    def __init__(self, name):
        self.name = name
        self.frames = 0
        self.obj_count = 0
        self.obj_score = 0
        self.bar = tqdm(desc="Progress")

    def frame(self):
        self.frames += 1
        self.bar.update(1)

    def update(self, score):
        self.obj_score += score
        self.obj_count += 1

    def report(self):
        print(f"\nPerformance report [{self.name}]")
        print(f"- Frames: {self.frames}")
        print(f"- Obj count: {self.obj_count}")
        print(f"- Avg precision: {self.obj_score / self.obj_count}")
