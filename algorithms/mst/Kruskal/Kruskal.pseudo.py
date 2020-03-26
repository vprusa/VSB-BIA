F ← ( V , ∅)
foreach v ∈ V ( G ) do MAKE-SET( v )
"sort edges according to w into a non-decreasing sequence"
foreach e = { u , v } in this order do
    if FIND-SET( u ) 6 = FIND-SET( v ) then
        F ← F ∪ e
        UNION( u, v )
