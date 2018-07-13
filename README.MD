#Day 25: Conjugate gradients
<br>
When you need to solve a system of linear equations, conjugate gradients present a fast way.
<br>
https://en.wikipedia.org/wiki/Conjugate_gradient_method
<br>
The idea is quite clever. Take a function F=L2(X’, X⁰)², that is a squared L2 norm of [not necessarily correct] solution X’ and the real solution X⁰. The function has parabolic shape and under the best scenario its minimum is located at origin (which would also be the solution). In such case, we may travel from X’ perpendicularly to each axis, one by one, to arrive into X⁰.
<br>
While this is not usually the case, real scenarios are not that much different. In contrast to the best scenario, conjugate gradients travel perpendicularly to eigenvectors which results in the same effect as described above.
<br>
Just remember that your matrix has to be positive semidefinite. If that’s not the case, use transformation (A’A)x=(A’b). That will handle overdetermined and underdetermined systems, too.