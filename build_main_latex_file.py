"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Main Article Skeleton Building
"""

from Solvers import *

header_file = "latex/header.tex"

footer_file = "latex/footer.tex"

image_comments = {'ackley': ["The isolines of Ackley Function and convergence lines for all algorithms "
                             "are shown in Figure~\\ref{fig:ackley}:",
                            "The isolines for Ackley Function shows that there are at least 4 pitfalls"
                            " near the optimal solution, so, the search space needs to be reduced to "
                            "$[-0.05, 0.05]$ to avoid wrong convergence. All the algorithms converges "
                            "quite smoothly following a straight line"],
                  "beale": ["The isolines of Beale Function and convergence lines for all algorithms "
                             "are shown i Figure~\\ref{fig:beale}:",
                            "The isolines for Beale Function shows that there are a large plateau near "
                            "the solution. For this reason we believe that the best algorithm had 80\% of performance"
                            " at its best. Quasi-Newton algorithms suffers to converge when the gradient is"
                            " less significant."],
                  "booth": ["The isolines of Booth Function and convergence lines for all algorithms "
                            "are shown i Figure~\\ref{fig:booth}:",
                            "The Figure~\\ref{fig:booth} shows that Booth Function has a well defined "
                            "valley near $[1, 3]$. "],
                  "matyas": ["The isolines of Matyas Function and convergence lines for all algorithms "
                             "are shown i Figure~\\ref{fig:matyas}:",
                             "The Figure~\\ref{fig:matyas} shows that Matyas Function has a well defined "
                            "valley near $[0, 0]$. "],
                  "rastrigin2d": ["The isolines of Rastrigin Function for 2 dimensions and "
                                  "convergence lines for all algorithms "
                                  "are shown in Figure~\\ref{fig:rastrigin2d}:",
                                   "The isolines for Rastrigin Function shows that there are at least 2 pitfalls"
                                   " near the optimal solution, so, the search space needs to be reduced to "
                                   "$[-0.15, 0.15]$ to avoid wrong convergence. All the algorithms, except BFGS, converges "
                                   "quite smoothly following a straight line"],
                  "rosenbrock2d": ["The isolines of Rosenbrock Function for 2 dimensions "
                                   "and convergence lines for all algorithms "
                                   "are shown in Figure~\\ref{fig:rosenbrock2d}:",
                                    "The isolines for Rosenbrock Function shows an interesting behaviour "
                                   " near the optimal solution: the banana shape seems to force all algorithms "
                                    "to choose $x_{2}$ axis to solve first, and then solve for $x_{1}$ axis. The $x_{2}$ axis "
                                    "has probably the most significant derivative for an arbitrary point, so, when the algorithm "
                                    "walks around for solution, it first find the valley and then walks around the valley to find its minumum"]}

content = dict()

content[2] = "\section{2D Function versions}\n\label{functions2D}\n\n"

content[4] = "\section{4D Function versions}\n\label{functions4D}\n\n"

content[30] = "\section{30D Function versions}\n\label{functions30D}\n\n"

for function in constraints:

    def_func = constraints[function]['def_func']

    defs = def_func()

    functionString = "{:s}{:d}D".format(function, defs['dimension'])

    functionName = function_names[function]

    content[defs['dimension']] += "\subsection{" + \
                                  functionName + \
                                  "}\n\label{" + \
                                  functionString + "}\n\n"

    content[
        defs['dimension']] += "\subsubsection{Convergence Analysis}\n\label{" + "convergence" + functionString + "}\n\n"

    with open("latex/" + function + "_convergence.tex", "r") as f:
        content[defs['dimension']] += f.read() + "\n"

    content[defs[
        'dimension']] += "\subsubsection{Statistical Analysis of The Solutions}\n\label{" + "statisticalanalysis" + functionString + "}\n\n"

    with open("latex/" + function + "_function_values.tex", "r") as f:
        content[defs['dimension']] += f.read() + "\n"

    content[defs['dimension']] += "\subsubsection{Best Fits}\n\label{" + "bestfits" + functionString + "}\n\n"

    with open("latex/" + function + "_best_fits.tex", "r") as f:
        content[defs['dimension']] += f.read() + "\n"

    with open("latex/" + function + "_best_solutions.tex", "r") as f:
        content[defs['dimension']] += f.read() + "\n"

    if defs['dimension'] == 2:
        # isolines

        content[defs[
            'dimension']] += "\subsubsection{Isolines and Convergence line}\n\label{" + "isolines" + functionString + "}\n\n"

        if function in image_comments:
            content[defs[
                'dimension']] += image_comments[function][0]



        content[defs['dimension']] += """\\begin{figure}[H]
\\centering
\\caption{Isolines and convergence line for """ + functionName + """}
\\label{fig:""" + function + """}
\\includegraphics[scale=0.5]{images/""" + function + """.jpg}
\\end{figure}
"""

        if function in image_comments:
            content[defs[
                'dimension']] += image_comments[function][1]

with open(header_file, "r") as f:
    header = f.read()

with open(footer_file, "r") as f:
    footer = f.read()

latex_content = header + "\n" + \
                content[2] + "\n" + \
                content[4] + "\n" + \
                content[30] + "\n" + footer

with open("paper/new/article.tex", "w") as f:
    f.write(latex_content)
