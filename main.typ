#import "template.typ" : *

#show: tud-slides.with(
  title: "Comparison of GPU programming models for computational fluid dynamics",
  organizational-unit: "The University of Queensland",
  authors: (
    (name: "Robert Watt", presenting: true),
    (name: "Peter Jacobs", presenting: false),
    (name: "Rowan Gollan", presenting: false),
    (name: "Shahzeb Imran", presenting: false)
  ),
  location-occasion: "Australasian Fluid Mechanics Conference",
  date: datetime(day: 2, month: 12, year: 2024)
)

// #title-slide

#slide(title: [The rise of GPUs])[
  // = The rise of GPUs

  Hello
]

#slide(title: [GPU programming models])[
  #grid(
    columns: (1fr, 1fr),
    inset: 4pt,
    [
      #rect(radius: 4pt, inset: 5pt, width: 100%,
        [
          == CUDA
            - Nvidia's language, for programming Nvidia GPUs
            - Based on C/C++ or Fortran
        ]
      )
      #rect(radius: 4pt, inset: 5pt, width: 100%,
        [
          == SYCL
            - Abstraction for heterogeneous programming in C++
            - Implemented by the compiler
            - Standard owned by Kronos group, a few implementations exist
        ]    
      )
      #rect(radius: 4pt, inset: 5pt, width: 100%,
        [
          == Others
            - RAJA, OCCA, ...
        ]
      )
    ],
    [
      #rect(radius: 4pt, inset: 5pt, width: 100%,
        [
          == HIP
            - AMD's language for programming AMD GPUs
            - Based on C/C++
            - Can also target Nvidia GPUs
        ]
      )
      #rect(radius: 4pt, inset: 5pt, width: 100%,
        [
          == Kokkos
            - C++ library providing abstractions for parallel programming
            - Originally developed by Sandia National Lab, now a Linux Foundation project
        ]
      )
      #rect(radius: 4pt, inset: 5pt, width: 100%,
        [
          == Chapel
            - Programming language from Cray/HP for high-performance computing
        ]
      )
    ]
  )
]

#slide(title: [Progamming models])[
  #grid(columns: (1fr, 1fr), inset: 4pt,
    [
      == CUDA
    ],
    [
      == Kokkos
    ]
  )
]

#slide(title: [The codes])[
  #grid(columns: (1fr, 1fr), inset: 4pt,
    [
      == Chicken
        - Block-structured grids
        - Finite volume
          - Selection of upwind flux caculators
          - 2nd order accuracy via MUSCL style reconstruction
          - Van-albada limiter
        - Viscous gradients calculated with least-squares
        - Time integration:
          - 3rd order Runge-Kutta time integration
    ],
    [
      == Ibis
        - Unstructured grids
        - Finite volume
          - Selection of upwind flux caculators
          - 2nd order accuracy via MUSCL style reconstruction
          - Barth-Jespersen limiter
        - Inviscid and viscous gradients calculated with least-squares
        - Time integration:
          - Any explicit Runge-Kutta
          - Jacobian-Free Newton-Krylov
    ]
  )
]

#slide(title: [The test problem])[
  #grid(columns: (2fr, 1fr), inset: 4pt,
  [
    Gaseous injection into Mach 4 cross-flow
  ],
  [
    == Numerics
      - 300 $times$ 300 $times$ 300 grid?
      - AUSMDV flux calculator
  ]
    
  )
]

#slide(title: [Code performance with grid resolution])[
  - Plot of the time per update per cell
]

#slide(title: [Acceleration on different hardware])[
  - Plot of acceleration of the different hardware
]

#slide(title: [GPU utilisation])[
  
]

#slide(title: [Conclusions])[
  - 
]

