class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        ans = None
        df = lambda x : 2*x
        curr = init
        while(iterations):
            curr = curr - learning_rate*df(curr)
            iterations -= 1 
        return round(curr,5)