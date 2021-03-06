using Documenter
using MEstimation

makedocs(
    sitename = "MEstimation",
    authors = "Ioannis Kosmidis, Nicola Lunardon",
    format = Documenter.HTML(),
    modules = [MEstimation],
    pages = [
        "Home" => "index.md",
        "Examples" => "man/examples.md",
        "Documentation" => Any[
            "Public" => "lib/public.md",
            "Internal" => "lib/internal.md",
        ]
    ],
    doctest = false
)

deploydocs(
    repo = "github.com/ikosmidis/MEstimation.jl.git",
    branch = "gh-pages",
    target = "build",
    devbranch = "develop",
    devurl = "dev",
    push_preview = true,
)

