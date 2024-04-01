from typing import List, Dict

# Type alias to refer to grammars later

Grammar = Dict[str, List[str]]

grammar = {
"_S" : ["_NP _VP"],
"_NP" : ["_N",
"_A _NP _P _A _N"],
"_VP" : ["_V",
"_V _NP"],
"_N" : ["data science", "Python", "regression"],
"_A" : ["big", "linear", "logistic"],
"_P" : ["about", "near"],
"_V" : ["learns", "trains", "tests", "is"]
}