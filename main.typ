#import "template.typ" : *
#import "@preview/nth:1.0.1": *

#show: tud-slides.with(
  title: "Comparison of GPU programming models for computational fluid dynamics",
  short-title: "Comparison of GPU programming models for CFD",
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

#slide(title: [The fragmentation of GPUs])[
  #grid(
    columns: (1fr, 1fr),
    [
      #set align(center)
      #text(weight: "bold", [Gadi:]) Nvidia
      #figure(
        image("h100.jpeg", height: 7em),
        caption: [https://www.nvidia.com/en-au/data-center/h100/]
      )
    ],
    [
      #set align(center)
      #text(weight: "bold", [Setonix:]) AMD Mi250X
      #figure(
        image("mi250x.png", height: 7em),
        caption: [https://www.nvidia.com/en-au/data-center/h100/]
      )
      
    ]
  )

  #uncover(2)[
    #grid(
      columns: (1fr, 1fr, 1fr), align: center,
      [
        #text(weight: "bold", [Frontier:]) AMD Mi250X
        #figure(
          image("mi250x.png", height: 7em),
          caption: [https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html]
        )
      ],
      [
        #text(weight: "bold", [El Capitan:]) AMD Mi300
        #figure(
          image("mi300x.png", height: 7em),
          caption: [https://www.amd.com/en/products/accelerators/instinct/mi300.html]
        )
      ],
      [
        #text(weight: "bold", [Aurora:]) Intel Data Center GPU Max
        #figure(
          image("intel_gpu.jpeg", height: 7em),
          caption: [https://www.tweaktown.com/news/89438/intels-new-data-center-gpu-max-1100-uses-controversial-12vhpwr-connector/index.html]
        )
      ]
    )
  ]
  // #uncover(2)[
  //   #set align(center + horizon)
  //   == Having the codes run on all GPUs is desirable
  // ]

]

#slide(title: [GPU programming models])[
  #grid(
    columns: (1fr, 1fr, 1fr),
    inset: 2pt,
    [
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == CUDA #box(height: 0.8em, image("cuda.svg"))
            #set text(size: 10pt)
            - Nvidia's language, for programming Nvidia GPUs
            - Based on C/C++ or Fortran
            - Runs on Nvidia GPUs
        ]
      )
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == #box(height: 0.8em, image("sycl.png"))
            #set text(size: 10pt)
            - Provides an abstraction for heterogeneous programming in C++
            - Implemented by various compilers (e.g. OneAPI by Intel)
            - Runs on any (supported) accelerator
        ]    
      )
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == #box(height: 0.8em, image("raja.png"))
            #set text(size: 10pt)
            - Similar to Kokkos, but split into more libraries
            - Runs on any (supported) accelerator
        ]
      )
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == OCCA #box(height: 0.8em, image("occa.svg"))
            #set text(size: 10pt)
            - Used in the first trillion degree of freedom simulation
            - Runs on any (supported) accelerator
        ]
      )
    ],
    [
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == ROCm/HIP #box(height: 0.8em, image("ROCm_logo.png"))
            #set text(size: 10pt)
            - AMD's language for programming AMD GPUs
            - Based on C/C++ or Fortran
            - Runs on AMD and Nvidia GPUs
        ]
      )
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == Kokkos #box(height: 0.8em, image("kokkos.png"))
            #set text(size: 10pt)
            - C++ library providing abstractions for parallel programming
            - Originally developed by Sandia National Lab, now a Linux Foundation project
            - Runs on any supported accelerator
            - Used in Trillinos
        ]
      )
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == Chapel #box(height: 0.8em, image("chapel-logo.png"))
            #set text(size: 10pt)
            - Programming language from Cray/HP for high-performance computing
            - Runs on CPUs, Nvidia GPUs, with some AMD and Intel GPU support
        ]
      )
    ],
    [
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == #box(height: 0.8em, image("OpenMP_logo.png"))
            #set text(size: 10pt)
              - Parallelism library built into C++ compilers
              - Allows for offload to many devices, including GPUs
        ]
      )
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == #box(height: 0.8em, image("OpenCL_logo.png"))
            #set text(size: 10pt)
            - Maybe dead? Maybe not?
        ]
      )
      #rect(radius: 4pt, inset: 4pt, width: 100%,
        [
          == #box(height: 0.8em, image("julia.svg"))
            #set text(size: 10pt)
            - Programming language with libraries to execute on Nvidia and AMD GPUs
        ]
      )
    ]
  )
]

#slide(title: [That's a lot of options...])[
  #set align(center + horizon)
  == Which programming model should we use?

  #uncover((beginning: 2))[
    == How well do they run across the different GPU brands?
  ]

  #h(2em)

  #uncover(3)[
    Todays talk: Compare Kokkos with pure CUDA
  ]
]

#slide(title: [Kokkos overview])[
  #figure(
    image("Kokkos-Architecture.png", height: 90%),
    caption: [Kokkos architecture (https://kokkos.org/about/abstract/)]
  )
]

#slide(title: [Kokkos overview])[
  - C++ library managing mapping of parallelism of the algorithm to the parallelism of the hardware
  - Can compile using vendor-provided compilers (e.g. CUDA or HIP), or can use SYCL, OpenMP, C++ threads etc.
  - Kokkos manages shared memory parallelisation
    - Compatible with MPI, HPX, PGAS etc.

  #rect(radius: 4pt, inset: 4pt,
    grid(
      columns: (1fr, 1fr), inset: 4pt,
      [
        == Advantages  
          - Automatic memory managment
          - Compile-time configurable memory layout (row major vs column major arrays)
            - Helps get the best memory layout for a particular piece of hardware
          - Transparent managment of different memory spaces
          - Can perform device specific optimisations
          - Free and open-source
      ],
      [
        == Disadvantages
          - Still C++
          - Relies heavily on template meta-programming
            - may be confusing for C++ beginners
            - Compile times are longer, but not necessarily prohibitive
      ]
    )
  )
]

#slide(title: [Example program])[
  #grid(columns: (1fr, 1fr), inset: 4pt,
    [
      == CUDA
    #set text(size: 10pt)
      ```cpp
      #ifdef HIP
      #define cudaMalloc hipMalloc
      #define cudaFree hipFree
      #endif

      __global__
      void initialise(double* a, int size) {
          int i = blockIdx.x*blockDim.x + threadIdx.x;
          if (i < size) {
              a[i] = 0.0;
          }
      }

      __global__
      void add_one(double* a, int size) {
          int i = blockIdx.x*blockDim.x + threadIdx.x;
          if (i < size) {
              a[i] += 1.0;
          }
      }

      int main() {
          // allocate an array of 10 doubles on the GPU
          double* a;
          cudaMalloc(&a, 10*sizeof(double));
          initialise<<<1, 32>>>(a, 10);
          cudaDeviceSyncronize();
          add_one<<<1, 32>>>(a, 10);
          cudaDeviceSyncronize();
          cudaFree(a);
      }
      ```
    ],
    [
      == Kokkos
      #set text(size: 10pt)
      ```cpp
      #include <Kokkos_Core.hpp>
      int main() {
        // allocate an array of 10 doubles
        // on the default device
        // (these will be zero initialised)
        Kokkos::View<double*, KokkosDefaultMemSpace> a(10);

        // Add one to each double in parallel
        Kokkos::parallel_for(10, KOKKOS_LAMBDA(const int i){
            a(i) += 1.0;
        });
      }
      ```

    ]
  )
]


#slide(title: [The codes])[
  #grid(columns: (1fr, 1fr), rows: 100%, inset: 4pt,
    [
      == Chicken #emoji.chicken
        - CUDA/HIP
        - #only((1,2,3))[Block-structured grids] #only((4,5))[*Block-structured grids*]
        - Finite volume
          - Selection of upwind flux caculators
          - 2nd order accuracy via MUSCL style reconstruction
          - Van-Albada limiter
        - Viscous gradients calculated with least-squares
        - Time integration:
          - 3rd order Runge-Kutta time integration
      #uncover((beginning: 2))[
        #set text(size: 13pt)
        #align(center+bottom,
          ```
               MM      --
              <' \___/| --
                \_  _/  --
                  ][   --
          -----------------
               CHICKEN
          -----------------
          ```
        )
      ]
    ],
    [
      #uncover((beginning: 3))[
        == Ibis #emoji.bin#emoji.chicken
          - Kokkos
          - #only(3)[Unstructured grids] #only((4,5))[*Unstructured grids*]
          - Finite volume
            - Selection of upwind flux caculators
            - 2nd order accuracy via MUSCL style reconstruction
            - Barth-Jespersen limiter
          - #only(3)[Inviscid] #only((4,5))[*Inviscid*] and viscous #only(3)[gradients] #only((4,5))[*gradients*] calculated with least-squares
          - Time integration:
            - Any explicit Runge-Kutta
            - Jacobian-Free Newton-Krylov        
      ]
    ]
  )
  #uncover((beginning: 5))[
    #set text(size: 13pt)
    #place(bottom+right, float: true, dx: -50pt, dy: -5pt,
      align(center, 
        ```
           MM
          <' \___/|
           \_  _/
             ][    
         ___/-\___
        |---------|
         | | | | |
         | | | | |
         | | | | |
         | | | | |
         |_______|
        ```
      )
    )
  ]
]

#slide(title: [The test problem])[
  #grid(columns: (1.3fr, 1fr), rows: 100%, inset: 4pt,
  [
    == The problem
      - Gaseous injection into Mach 4 cross flow

    #set align(horizon)
    #figure(
      image("domain_schematic_2.svg"),
      caption: [Domain and boundary conditions]
    )<domain-schematic>
  ],
  [
    == Conditions
      #set text(size: 14pt)
      #set align(center)
      #table(
        columns: (auto, auto, auto, auto),
        inset: 5pt,
        stroke: none,
        align: horizon+center,
        table.hline(),
        table.header([], [Pressure \ (kPa)], [Temperature \ (K)], [Velocity \ (m/s)]),
        table.hline(),
        [Free-stream], [1.013], [300], [1390],
        [Injector], [10.013], [300], [350],
        table.hline(),
      )
    #set text(size: 15pt)
    #set align(left)
    == Grid
      - Various grid resolutions from 0.5-3.5~million cells
      - `Chicken` used blocking structure indicated in @domain-schematic
      - `Ibis` merged all blocks into one block

    == Numerics
      - AUSMDV flux calculator
      - #nth(2) order accurate spatially
      - #nth(3) order Runge-Kutta time-stepping
  ]
    
  )
]

#slide(title: [The test problem])[
  #figure(
    image("visualisation.png", height: 80%),
    caption: [Flow field after 5ms]
  )
]

#slide(title: [Code performance with grid resolution])[
  #set text(size: 14.5pt)
  #grid(
    columns: (1.4fr, 1fr), align: horizon,
    [
      #figure(
        image("h100_performance.svg", width: 100%),
        caption: [Time to update a cell at various grid resolutions on an Nvidia H100]
      )
    ],
    [
      == Motivation
        - Check to make sure the GPU is saturated
          - `Chicken` may require more cells to saturate the GPU
        - Get base-line performance of each code

      == Reasons for differences
        - `Chicken` moves memory for #nth(2) order around even when doing #nth(1) order, so #nth(1) order is slow
        - `Chicken` calculates all fluxes in one large kernel
          - This may be causing register pressure, leading to lower occupancy than `Ibis`
          - The kernel with the lowest occupancy in `Ibis` had twice the occupancy of `Chicken`'s fused kernel
        - `Chicken` uses array-of-structures, `Ibis` uses structure-of-arrays
    ]
  )
]

#slide(title: [Acceleration on different hardware])[
  #grid(
    columns: (1.5fr, 1fr), align: horizon,
    [
      #figure(
        image("gpu_acceleration.svg", width: 100%),
        caption: [Acceleration compared to serial CPU performance on various GPUs]
      )
    ],
    [
      - `Ibis` (Kokkos) had greater acceleration on all HPC grade GPUs
        - Likely due to `Ibis` having greater occupancy owing to smaller kernels, smaller register count, higher occupancy, and better memory access patters
      - `Chicken` (CUDA) had greater acceleration on consumer grade GPU
        - Likely due to few FP64 cores on consumer grade GPUs
    ]
  )
]

#slide(title: [GPU utilisation])[
  #grid(columns: (1.5fr, 1fr), align: horizon,
    [
      #figure(
        image("roofline.svg", width: 100%),
        caption: [Roofline plot of the two codes on an Nvidia A100. Size of the circles represents time spent in that kernel]
      )
    ],
    [
      - Closer to the black bline is better (above is impossible)
      - Some long-running kernels in `Ibis` benefit from being their own kernel with lower register pressure
      - `Ibis` does more floating point operations per byte of data transferred partially due to better memory layout.
    ]
  
  )
]

#slide(title: [Conclusions])[
  - Writing a code in CUDA doesn't mean its fast
    - Leave the details of the parallelism to the experts, and we'll write the physics
  - Code design and memory access patterns had a larger impact on performance than overhead from abstractions
  - Kokkos seems to be a good option for performance portable code
  - But maybe one of the others is better?

  #place(bottom+center, float: true, dy: -50pt,
    grid(columns: (1fr, 1fr),
    [
      ```
         MM
        <' \___/|
         \_  _/
           ][    
       ___/-\___
      |---------|
       | | | | |
       | | | | |
       | | | | |
       | | | | |
       |_______|
      ```
    ],
    [
      ```
           MM      --
          <' \___/| --
            \_  _/  --
              ][   --
      -----------------
           CHICKEN
      -----------------
      ```
      
    ]
      
    )
  )
]

