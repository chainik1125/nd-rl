## Elicitation vs Emergence

I think there are many precise and mathematical ways to define this which are rigorous and somewhat
boring, eg
https://github.com/dan-pandori/dan-pandori-blog-reports/tree/main/posts/elicitation-vs-emergence.

I am more excited by definitions which correspond to interesting qualitative phenomena. A guiding
light: normies say shit like "Astra didn't invent new math, it just read all the best papers and
made a couple logical leaps that no one had had the time to make before, and this was all it took to
prove {result}. I think this complaint is reasonable when we discuss a calculator adding to 50
digits numbers which have never before been added, and unreasonable when we discuss Navier-Stokes.
If we had some way to draw a bright line inbetween, that would be very interesting. But it sounds
very challenging.

I think we've seen some small examples of this in our toy setting (eg models learning or not
learning proof by contradiction). But also in some sense Natural-Deduction feels anti-optimized for
this, because the setting is so constrained.

I do not have a good answer to the question at all, but it was worth me spending five minutes naming
my confusion.

Throwing some ideas at the wall:

- emergence if a dumb model never learns with infinite compute / RL, and a smart model does
- emergence if it is impossible to pretrain a model to do the thing, but a model can _create_ its
  own posttraining data and and train on it to learn the thing, eg there needed to be 'intervening
  learning' in the form of new data created and trained on, like some evolutionary process
- possibly there is some way to coerce EDL into looking like 'qualitatively different'?

I am not so pesimistic about formalizing the above?

(Aside: maybe real capabilities emergence would be models learning Dan's hack of a 16-way proof
table. It is in principle possible to structure any proof like this? Assume P,Q,R,S, and ...).
