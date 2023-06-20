class Plant:
    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def is_burning(self):
        return self.state == 2

    def is_tree(self):
        return self.state == 1

    def is_empty(self):
        return self.state == 0

    def __repr__(self) -> str:
        return f"{self.state}"