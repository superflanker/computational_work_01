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

    content[defs['dimension']] += "\nIntroductory text goes here...\n"

    with open("latex/" + function + "_convergence.tex", "r") as f:
        content[defs['dimension']] += f.read() + "\n"

    content[defs['dimension']] += "\nComments goes here...\n"

    content[defs[
        'dimension']] += "\subsubsection{Statistical Analysis of The Solutions}\n\label{" + "statisticalanalysis" + functionString + "}\n\n"

    content[defs['dimension']] += "\nIntroductory text goes here...\n"

    with open("latex/" + function + "_function_values.tex", "r") as f:
        content[defs['dimension']] += f.read() + "\n"

    content[defs['dimension']] += "\nComments goes here...\n"

    content[defs['dimension']] += "\subsubsection{Best Fits}\n\label{" + "bestfits" + functionString + "}\n\n"

    content[defs['dimension']] += "\nIntroductory text goes here...\n"

    with open("latex/" + function + "_best_fits.tex", "r") as f:
        content[defs['dimension']] += f.read() + "\n"

    content[defs['dimension']] += "\nComments goes here...\n"

    with open("latex/" + function + "_best_solutions.tex", "r") as f:
        content[defs['dimension']] += f.read() + "\n"

    content[defs['dimension']] += "\nComments goes here...\n"

    if defs['dimension'] == 2:
        # isolines

        content[defs[
            'dimension']] += "\subsubsection{Isolines and Convergence line}\n\label{" + "isolines" + functionString + "}\n\n"

        content[defs['dimension']] += "\nIntroductory text goes here...\n"

        content[defs['dimension']] += """\\begin{figure}[h]
\\centering
\\includegraphics[scale=0.5]{images/""" + function + """.jpg}
\\caption{Isolines and convergence line for """ + functionName + """}
\\label{fig:""" + function + """}
\\end{figure}
"""

        content[defs['dimension']] += "\nComments goes here...\n"

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
