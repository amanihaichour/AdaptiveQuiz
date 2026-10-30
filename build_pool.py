import json
import random

# Generate a pool of highly distinct math questions scaling strictly with difficulty
questions = []

algebra_q = [
    (0.1, "Solve for x: x + 2 = 5", {"A":"2", "B":"3", "C":"4", "D":"5"}, "B"),
    (0.2, "Solve for x: 3x = 12", {"A":"2", "B":"3", "C":"4", "D":"5"}, "C"),
    (0.3, "Solve for x: 2x - 5 = 9", {"A":"7", "B":"6", "C":"8", "D":"5"}, "A"),
    (0.4, "Expand: (x + 3)(x - 2)", {"A":"x^2+x-6", "B":"x^2-x-6", "C":"x^2-6", "D":"x^2+5x-6"}, "A"),
    (0.5, "Solve the system: x+y=10, x-y=2", {"A":"x=6,y=4", "B":"x=4,y=6", "C":"x=5,y=5", "D":"x=8,y=2"}, "A"),
    (0.6, "Factor completely: x^2 - 8x + 15", {"A":"(x-3)(x-5)", "B":"(x+3)(x-5)", "C":"(x-3)(x+5)", "D":"(x+3)(x+5)"}, "A"),
    (0.7, "Solve for x: log2(x) + log2(x-3) = 2", {"A":"x=4", "B":"x=1,4", "C":"x=4,-1", "D":"x=2"}, "A"),
    (0.8, "Find the roots of: x^3 - 6x^2 + 11x - 6 = 0", {"A":"1,2,3", "B":"-1,-2,-3", "C":"1,-2,3", "D":"-1,2,-3"}, "A"),
    (0.9, "Solve the matrix equation det([[x, 2], [3, x-1]]) = 0", {"A":"3, -2", "B":"-3, 2", "C":"3, 2", "D":"-3, -2"}, "A"),
    (0.95, "Find the eigenvalues of [[2, 1], [1, 2]]", {"A":"1, 3", "B":"2, 2", "C":"0, 4", "D":"1, 2"}, "A")
]

calculus_q = [
    (0.1, "Find the derivative of f(x) = 5x", {"A":"5", "B":"x", "C":"5x", "D":"0"}, "A"),
    (0.2, "Find the derivative of f(x) = x^2", {"A":"2x", "B":"x", "C":"2", "D":"x^2"}, "A"),
    (0.3, "Evaluate the integral of 2x dx", {"A":"x^2 + C", "B":"2x^2 + C", "C":"x^2", "D":"2 + C"}, "A"),
    (0.4, "Find the derivative of sin(x)", {"A":"cos(x)", "B":"-cos(x)", "C":"-sin(x)", "D":"sec(x)"}, "A"),
    (0.5, "Evaluate the limit as x -> 0 of sin(x)/x", {"A":"1", "B":"0", "C":"infinity", "D":"undefined"}, "A"),
    (0.6, "Find the derivative of e^(2x)", {"A":"2e^(2x)", "B":"e^(2x)", "C":"2xe^(2x)", "D":"e^x"}, "A"),
    (0.7, "Evaluate the definite integral of x^2 from 0 to 3", {"A":"9", "B":"3", "C":"27", "D":"6"}, "A"),
    (0.8, "Find the derivative of x * ln(x)", {"A":"ln(x) + 1", "B":"1/x", "C":"ln(x)", "D":"x/ln(x)"}, "A"),
    (0.9, "Evaluate the integral of e^x * sin(x) dx", {"A":"(e^x(sin x - cos x))/2 + C", "B":"e^x sin x + C", "C":"(e^x(sin x + cos x))/2 + C", "D":"-e^x cos x + C"}, "A"),
    (0.95, "Find the Maclaurin series expansion of e^x up to x^3", {"A":"1 + x + x^2/2 + x^3/6", "B":"1 - x + x^2/2 - x^3/6", "C":"x + x^2/2 + x^3/3", "D":"1 + x + x^2 + x^3"}, "A")
]

probability_q = [
    (0.1, "Probability of flipping heads on a fair coin", {"A":"0.5", "B":"1.0", "C":"0.25", "D":"0.0"}, "A"),
    (0.2, "Probability of rolling a 4 on a standard 6-sided die", {"A":"1/6", "B":"1/4", "C":"1/2", "D":"1/3"}, "A"),
    (0.3, "Probability of drawing a spade from a standard 52-card deck", {"A":"1/4", "B":"1/13", "C":"1/52", "D":"1/2"}, "A"),
    (0.4, "If P(A)=0.4 and P(B)=0.3 (independent), find P(A and B)", {"A":"0.12", "B":"0.7", "C":"0.1", "D":"0.0"}, "A"),
    (0.5, "Expected value of a fair 6-sided die roll", {"A":"3.5", "B":"3", "C":"4", "D":"3.16"}, "A"),
    (0.6, "How many ways can you arrange 4 different books on a shelf?", {"A":"24", "B":"16", "C":"4", "D":"12"}, "A"),
    (0.7, "What is '5 choose 2' (Combinations)?", {"A":"10", "B":"20", "C":"120", "D":"5"}, "A"),
    (0.8, "Probability of getting exactly 2 heads in 3 coin flips", {"A":"3/8", "B":"1/8", "C":"1/2", "D":"1/4"}, "A"),
    (0.9, "Bayes Theorem: If P(D)=0.01, P(Pos|D)=0.9, P(Pos|~D)=0.1, find P(D|Pos)", {"A":"0.083", "B":"0.9", "C":"0.01", "D":"0.5"}, "A"),
    (0.95, "What is the variance of a Binomial distribution with n=10, p=0.4?", {"A":"2.4", "B":"4.0", "C":"6.0", "D":"2.0"}, "A")
]

statistics_q = [
    (0.1, "Find the mean of 2, 4, 6", {"A":"4", "B":"3", "C":"6", "D":"12"}, "A"),
    (0.2, "Find the median of 1, 9, 3, 5, 7", {"A":"5", "B":"3", "C":"9", "D":"6"}, "A"),
    (0.3, "Find the mode of 2, 3, 3, 4, 5", {"A":"3", "B":"2", "C":"4", "D":"5"}, "A"),
    (0.4, "If variance is 25, what is the standard deviation?", {"A":"5", "B":"12.5", "C":"625", "D":"2.5"}, "A"),
    (0.5, "In a normal distribution, what % of data falls within 1 standard dev?", {"A":"68%", "B":"95%", "C":"99%", "D":"50%"}, "A"),
    (0.6, "What is a Type I error?", {"A":"False Positive", "B":"False Negative", "C":"True Positive", "D":"True Negative"}, "A"),
    (0.7, "Calculate the Z-score for X=85 if mean=70 and std=5", {"A":"3.0", "B":"2.0", "C":"1.5", "D":"15"}, "A"),
    (0.8, "What does a p-value of 0.01 indicate (alpha=0.05)?", {"A":"Reject Null Hypothesis", "B":"Accept Null Hypothesis", "C":"Inconclusive", "D":"Fail to reject Null"}, "A"),
    (0.9, "What is the standard error of the mean for std=10, n=25?", {"A":"2.0", "B":"0.4", "C":"2.5", "D":"10.0"}, "A"),
    (0.95, "Which test compares the means of three or more independent groups?", {"A":"ANOVA", "B":"T-test", "C":"Chi-Square", "D":"Pearson"}, "A")
]

geometry_q = [
    (0.1, "Area of a rectangle with length 5 and width 4", {"A":"20", "B":"9", "C":"18", "D":"40"}, "A"),
    (0.2, "Perimeter of a square with side length 3", {"A":"12", "B":"9", "C":"6", "D":"15"}, "A"),
    (0.3, "Sum of interior angles of a triangle", {"A":"180", "B":"360", "C":"90", "D":"270"}, "A"),
    (0.4, "Area of a circle with radius 3", {"A":"9pi", "B":"6pi", "C":"3pi", "D":"12pi"}, "A"),
    (0.5, "Hypotenuse of a right triangle with legs 3 and 4", {"A":"5", "B":"6", "C":"7", "D":"25"}, "A"),
    (0.6, "Volume of a cube with side length 2", {"A":"8", "B":"4", "C":"6", "D":"12"}, "A"),
    (0.7, "Circumference of a circle with diameter 10", {"A":"10pi", "B":"5pi", "C":"25pi", "D":"100pi"}, "A"),
    (0.8, "Sum of interior angles of a hexagon", {"A":"720", "B":"540", "C":"360", "D":"900"}, "A"),
    (0.9, "Distance between points (1,2) and (4,6)", {"A":"5", "B":"3", "C":"4", "D":"25"}, "A"),
    (0.95, "Equation of a circle centered at (0,0) with radius 4", {"A":"x^2 + y^2 = 16", "B":"x^2 + y^2 = 4", "C":"x + y = 16", "D":"x^2 + y^2 = 8"}, "A")
]

# Build the JSON
out = []
q_id = 1
for topic_name, q_list in [("algebra", algebra_q), ("calculus", calculus_q), ("probability", probability_q), ("statistics", statistics_q), ("geometry", geometry_q)]:
    # Let's shuffle options to make it realistic
    for diff, text, opts, corr in q_list:
        letters = ["A", "B", "C", "D"]
        items = list(opts.items())
        random.shuffle(items)
        
        new_opts = {}
        new_corr = "A"
        for i, (old_letter, val) in enumerate(items):
            new_opts[letters[i]] = val
            if old_letter == corr:
                new_corr = letters[i]
                
        out.append({
            "id": f"q{q_id}",
            "topic": topic_name,
            "difficulty": diff,
            "text": text,
            "options": new_opts,
            "correct": new_corr,
            "explanation": f"The correct answer is {new_opts[new_corr]}."
        })
        q_id += 1

with open("src/ui_questions.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("Created highly balanced and scaled questions!")
