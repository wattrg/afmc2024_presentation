#import "@preview/polylux:0.3.1": polylux-slide, logic, utils, pause, only
// #import "@preview/touying:0.4.2": *

#let tud-outer-margin = 30pt
#let tud-inner-margin = 30pt
#let tud-top-margin = 60pt
#let tud-bottom-margin = 40pt

#let tud-title = state("tud-title", none)
#let tud-subtitle = state("tud-sub-title", none)
#let tud-short-title = state("tud-short-title", none)
#let tud-authors = state("tud-authors", [])
#let tud-short-author = state("tud-short-author", none)
#let tud-organizational-unit = state("tud-organizational-unit", none)
#let tud-date = state("tud-date", none)
#let tud-location-occasion = state("tud-location-occasion", none)

#let uq-purple = rgb(81, 36, 122)
#let uq-white = rgb(121, 76, 162)
#let uq-gradient = gradient.linear(uq-purple, uq-white, angle: 45deg)

#let tud-show-guides = state("tud-show-guides", none)
#let tud-date-format = "[month repr:long] [day], [year]"

#let title-slide(title: none, authors: none) = {
  let content = {
    place(
      rect(fill: uq-gradient, width: 100%, height: 100%)
    )

    block(
      width: 100%,
      fill: white,
      inset: (x: tud-outer-margin, y: 0.8cm),
      grid(
        columns: (1fr, auto),
        image("logo_alt.png", height: 1.33cm),
        // image("lo", height: 1.33cm)
      )
    )

    place(
      block(
        width: 100%,
        height: 85%,
        inset: (x: tud-inner-margin),
        {
          set align(left + horizon)
          linebreak()
          text(size: 2.5em, fill: white, weight: "bold", context tud-title.get())
          parbreak()
          context tud-subtitle.get()
          parbreak()
          set text(fill: rgb(255, 255, 255, 200))
          let n-authors = authors.len()
          for (i, author) in authors.enumerate() {
            if author.presenting [
              #text(weight: "bold", author.name)
            ] else [
              #author.name
            ]
            if i < n-authors - 2 [, ] else if i < n-authors - 1 [, and]
          }
          linebreak()
          set text(fill: rgb(255, 255, 255, 200))
          text(
            weight: "bold",
            context tud-organizational-unit.get()
          )
          linebreak()
          linebreak()
          context [#tud-location-occasion.get()]
          linebreak()
          context [#tud-date.get().display(tud-date-format)]
        }
      )
    )
  }

  polylux-slide(content)
}

#let tud-slides(
  title: none,
  subtitle: none,
  short-title: none,
  authors: (),
  organizational-unit: none,
  lang: "en",
  date: datetime.today(),
  location-occasion: none,
  show-guides: false,
  body
) = {
  set document(title: title)

  let guides = {
    set line(
      stroke: (thickness: .5pt,
      paint: red,
      dash: "densely-dotted"),
      length: 100%
    )
    place(line(start: (0cm, .8cm)))
    place(line(start: (0cm, 60pt)))
    place(line(start: (0cm, 2.92cm)))
    place(line(start: (0cm, 14.23cm)))
    place(line(start: (0cm, 14.933cm)))
    place(line(start: (0cm, 15.424cm)))
    place(line(start: (0cm, 16.22cm)))
    set line(angle: 90deg)
    place(line(start: (60.5pt, 0cm)))
    place(line(start: (790pt, 0cm)))
  }

    set page(
      // seems to correspond to 297mm × 167mm (as in the indesign template)
      paper: "presentation-16-9",
      margin: 0pt,
      foreground: context if (tud-show-guides.get() == true) {guides},
    )
    set text(font: ("Open Sans"), lang: lang, fill: uq-purple)
    // set list(marker: [---])

    show figure.caption: set text(size: .6em, fill: luma(150))

    tud-date.update(date)
    tud-title.update(title)
    tud-subtitle.update(subtitle)
    tud-short-title.update(title)
    tud-organizational-unit.update(organizational-unit)
    tud-location-occasion.update(location-occasion)
    tud-show-guides.update(show-guides)

    title-slide(title: title, authors: authors)

    body
  }

  #let footer = block(width: 100%, height: 100%, fill: uq-purple)[
    #set text(size: 8pt,  fill: white)
    #block(width: 100%, inset: (x: tud-outer-margin, top: 10pt),
      align(
        horizon,
        grid(
          columns: (1fr, 1fr),
          // gutter: 50pt,
          image(
            "uq-logo-white.svg", height: 22pt
          ),
          align(right,
            [
              Slide #logic.logical-slide.display()/#strong(utils.last-slide-number)
            ],        
          )
        )
      )
    )
  ]

  #let slide(title: none, body) = {
    // Content block
    let wrapped-body = block(
      width: 100%,
      height: 100%,
      inset: (x: tud-inner-margin),
    )[
      #set text(16pt, fill: black)
      #set block(above: 1.2em)
      #body
    ]

    let header = {
      block(width: 100%, height: 100%, fill: uq-purple)[
          #block(width: 100%, height: 100%, inset: (x: tud-outer-margin, top: 0pt))[
            #set align(top)
            #set align(horizon)
            #show heading: set text(fill: white, size: 2em)
            #heading(title)
            
          ]
      ]
    }

    set page(
      margin: (
        top: tud-top-margin,
        bottom: tud-bottom-margin
      ),
      header: header,
      footer: footer,
      footer-descent: 0pt,
    )
    polylux-slide(wrapped-body)
  }

  #let fluid-slide(body) = {
    // Content block
    let wrapped-body = block(
      width: 100%,
      height: 100%,
    )[
      #set text(16pt)
      #set block(above: 1.2em)
      #body
    ]

    set page(
      margin: (
        bottom: tud-bottom-margin
      ),
      footer: footer,
      footer-descent: 0pt,
    )
    polylux-slide(wrapped-body)
  }

  #let section-slide(title: none, subtitle: none) = {
    // Content block
    let wrapped-body = block(
      width: 100%,
      height: 100%,
      inset: (x: tud-inner-margin),
    )[
      #set align(horizon)
      #set text(2.5em, fill: white)
      #set block(above: .5em)
      #text(weight: "bold", title)
      #parbreak()
      #text(fill: rgb(255, 255, 255, 150), subtitle)
    ]

    set page(
      margin: (
        top: tud-top-margin,
        bottom: tud-bottom-margin
      ),
      footer: footer,
      footer-descent: 0pt,
      background: {
        rect(width: 100%, height: 100%, fill: uq-gradient)
      },
    )
    polylux-slide(wrapped-body)
}
