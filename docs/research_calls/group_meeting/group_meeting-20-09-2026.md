NDRL Group meeting - September 20
VIEW RECORDING - 64 mins (No highlights): https://fathom.video/share/XTdagTJajBRTMxPAVqQVfxa1CDi-a36-

---

0:02 - Dan Pandori
  Yeah, the reason for that was much more that I couldn't hack the commute than me not thinking that AI safety was the biggest deal, but conveniently now my wife is retired and I can move up to Berkeley if, when, you know, I get a job up there.  So that's the thought for the next time. Okay, okay, huge.

0:24 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Charles, do you want to go next? Sorry, I should have seen that you were chomping there when I left you.

0:32 - Charles Yu
  No, you're good, you're good. Yeah, so I'm Charles.

0:35 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I haven't done any AI safety work before, but I studied like math and computer science in college.

0:40 - Charles Yu
  I just graduated, so coming in with like a fresh mind, yeah. Cool. I can go next.

0:50 - Leon Lenk
  I, my name is Leon. Um, my AI safety experience is mostly just going to, uh, like a, my club at the university.  We did like a lot of reading groups, so I'm pretty familiar with papers, but not like that much technical experience.  My name's Robbie. I live in San Francisco.

1:14 - Robert Thompson
  I have not that much safety experience either. I've been reading What's Wrong for a year or two, and I've done a couple of side projects on my own, but this is my first big gander.  Cool.

1:26 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Okay, terrific. Yeah, always easier to do these kinds of things in person, but we do what we can. Cool.  All right, so this is our first all-group meeting. I think the hope for these meetings in general is that we will present results based on the stuff that we'd agreed to do last week, have a think about why they're good, why they're bad, what we want to keep, what we want to get rid of, and then set goals for the meeting next week.  And so... . So, you know, the good way for me to have run this meeting would be I would present the results that I have from the previous times I've tried to work on this project and said what things are going forward.  But, yeah, iClear has not been kind to the project deadline. So I just want to talk about kind of different approaches to what we can do at a high level, the different phases of the project as I see them, and then think about what people can do going forward.  So just as a logistics thing, I'm still waiting for my university to tell SPA that it's okay for them to give me money for project expenses.  Once that goes through, what will happen is SPA will give me an expense card. I will give that expense card to you on the understanding that, you know, you won't use it for nefarious things and also you won't tell SPA that I gave you the card directly.  And... That you will give me the invoices and receipts for all of the expenses that you do. And I think it'll just be easier that way.  And then once you have that, you you should at the least purchase yourself, you know, whatever the max plan is for whichever of Codex or Claude you prefer.  If you find yourself running out of credits, you should buy yourself another Claude or Codex $200 a month thing.  I think we can probably support up to like three max plans per person. But obviously, we don't want to do that unless we have to.  So if you find yourself running out of credits, let me know. We'll probably just do another one. If you find yourself still running out of credits after like three Claude maxes, then I might have to bug spar for more money.  Quick logistic question on just how the budget of that works out.

4:05 - Dan Pandori
  If there's five of us and the expectation is there's like 10k total, a $200 a month plan for five of us is running $1,000 a month.  This route is going to run for four months. So you're basically saying you're expecting that to be like 40% of the 10k budget is like one cloud next plan?  Yeah, think the the thing is compute is fairly cheap.

4:27 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  don't expect us to. Yeah, I don't expect. So roughly if I think about 100 days, $10 a $10 a day of compute per person, that's roughly like half a day's worth of H100 hours.  Then that kind of gets me to the project balance. Okay, thanks.

4:56 - Dan Pandori
  Yeah, I think also like, yes.

5:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  We have $10k right now. If we need more money, then we can get more money. Subject, obviously, to making decent progress.  Any other questions about that? Good. Yeah, just, you I think I've said this before, but I'll say it again just in case Charles needs to hear it.  I expect this project is going to be extremely difficult to do without Claude Code, so you should use the agents aggressively for most of the stuff that you do, and that's like a good research skill that you need to build.  Cool. Alright, so I want to now pivot to talking a little bit about Phase of the Project. So I basically...  We view the project as being in three phases. Each of them is roughly a month and then there's a month slack time.  So the first phase is we want to basically do very scrappy signs of lifestyle experiments on trying to get to a feel of a result that we feel happy when we get down to writing the paper.  However, if the more principled way of doing what we do goes the way we expect, then that will end up being a good paper.  That's kind of the goal of this first phase. And I think the rough benchmark that you should have in this first phase is that it should take you somewhere between three and seven days, maybe three and five days of full time time to go from I don't know what I'm doing to I tried this idea and here is the two or three graphs that tell me.  be me. Either why I'm excited about it or either why I think this is a dead end and we shouldn't pursue it anymore.  So broadly, the way I expect the workflow that will be most effective for all of you to go is you create a branch like Dan, experiment, blah, blah, blah.  You run some experiment, you have some proposal doc, you review that proposal after it's done and then you put it either into the stuff we want to keep, archived experiments folder of the .md or stuff that's dead and we don't want to think about any more part of the repo.  And, you know, maybe as time goes on we'll resurrect stuff from the graveyard, but more or less I kind of think that you basically just need to go with whatever feels like it's working the best.  I think the sort of rough heuristic for AI safety is that it's a fairly new field, it's growing fairly quickly.  The if stuff. . That's kind of like the rough vibe that we want to have in the first month.  Has anyone read the Ethan Perez post on this? What is the name of the post? It's like tooling up in AI safety or something like that.  Let me see if I can find it. Oh, here we go. Tips and code for are empirical research workflows.  So most of this is now out of date on a technical level, so I don't encourage you to really use any of the stuff that they say they're beyond stuff like Tmux, etc.  But for the general tools that you want to have for operating very quickly to get to a decent either yay or nay on an idea, I think it's pretty good.  Cool. All right. So that's phase one. Success would be at the end of phase one. So on October 20th, we have our Sunday meeting and we say, okay, the vibe of the paper that we want to write is broadly something like, hey, we showed that here is this crossover that you get between elicitation and new capability emergence in RL.  So we want to see, does that crossover hold both in RL and RL? That would be an example of the kind of hypothesis I would hope we would arrive at by the end of the first month with some initial signs of life experiments that are like, oh, this seems broadly plausible.  That's phase one. Uh, any questions about that? Cool. Uh, phase two, uh, is when we'll start to build uh, like actual infrastructure that we care about for carrying out the proposal that we have in, uh, phase one escape.  So, uh, an ideal outcome here is like, I basically expect that we will never merge, um, we'll never merge people's branches in Git until the end of the first month.  And then the Git repo will be a complete mess with probably hundreds of different work trees, lots of different branches.  And if you've ever done proper software engineering before, it should be anathema to you. But that's fine. After we're settled on what we feel like is going to be the hypothesis that's gonna work, then we'll build some infrastructure around that to do that at scale, which will support reasonable research practices like, hey, let's try this in a couple of different models, let's try this in a couple of different contexts, let's quantitatively measure the effect that we have between different seeds, et cetera, et cetera.  But none of that stuff I think is worth doing until we have the signs of life that we have in Phase I.  So for example, to go with the earlier example in Phase II, if we're like, okay, we think there's this crossover in elicitation versus capability emergence, then what we probably want to do is do a really good job at different and talk  If we find that there's another setting which could be insightful, a different setting which is better, then we should do the same thing for that.  We should have a panel of models on which we consider it, and then we should be very careful about using the RL implementations for the stuff that we're considering that's consistent with the literature.  At this phase, it's possible that we'll find, okay, the idea which seemed promising in month one is no longer promising in month two, because when we really looked at it, we figured out that we were deceived by some experimental artifact, and that would be unfortunate, but such is life, and we may have to pivot at that point.  But I think the probability of that happening will basically be a function of how well we do in month one.  Month three is then cleaning up the stuff that we've done before and preparing ourselves to submit to a conference.  So... I think a good target for what we're doing now is to try to submit a workshop paper to iClear or something like that.  If things are going really well, we can just go ahead and submit a full conference paper. I've done that before for SPAR projects, but I think that's totally dependent on the results.  I think, though, it's good to have a rough heuristic in mind of like, okay, I should at least be able to defend the current state of the results, the level of workshop paper by the end of the three months.  Cool. Questions, concerns, comments about those three phases before I talk about the first phase in a little bit more detail.  None. Terrific. Okay. Yeah. Yeah. All right, so what we're going to be doing this month. So, I think that we are basically doing two things in parallel.  One, we are trying to test out whatever we think is our current best idea in our current setting. Two, I think we should be extremely skeptical as we're going through this that what we're doing is the right thing to be doing.  So, you should have in the back of your mind good questions like, what is natural deduction buying us over and above what people have already demonstrated in, say, AlphaGo?  You should be thinking, why are we doing a project on RL capability emergence if we already know that that's possible from things like robotics?  You should be asking yourself if what we care about is a mechanistic understanding of how RL installs capabilities, why would  We expect to get anything different from, for example, the how reasoning models, base models know how to think, reasoning models know when, or other mechanistic literature which has tried to attack this question before.  So I think a lot of this month is trying to like more refine what precisely is the question that we're talking about that's still interesting but still solvable.  Okay. So, how does that look like? In week one, I would like you all to try to come up with your own version of an auto research pipeline that you use to hill climb on proof length from a given set of, from a fixed initial data set, you know, in the, in the like take home, what I asked you to do was like fix that length to length six.  right. right. right. All Just because that was computationally feasible, and then scale and see how far you can scale.  Beyond that, that doesn't have to be like how you draw the boundary. You should draw the boundary in the way that you consider to be principled, but I think that in this week it would be good for you all to see what day-to-day pipeline of using the agents is effective for you, and how good does it do on the tasks that we care about.  This will pay dividends for you and for the project, for the rest of the project, so I think this is a good place to start.  Where are people at with that? How far along is each person on implementing some auto-research thing based loosely on the literature we talked about on Thursday?

16:53 - Dan Pandori
  I've got something spun up. I'm deeply ashamed of it from a software engineering standpoint, but like, you know, it- ...

16:59 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  ... can run things and it seems to have created some interesting experiments that I need to do a better job reading.

17:05 - Dan Pandori
  Currently the bottleneck for me is by far understanding the results that the models produce rather than directing the models.  Yeah, I think that's broadly what I expect.

17:18 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  think that there is a very difficult thing of like how much review is worth your time given how much results the models produce.  Can you give us the parameters of the pipeline? Like what does it roughly look like?

17:32 - Dan Pandori
  Yeah, so I have DigitalOcean running a couple of persistent VPS for the agents to use my cloud code to actually spin off additional experiments.  I have a RunPod account that they also have access to that they can run GPUs on, know, download the repo, all that sort of thing.  There is a HuggingFace project. There I forget what they call them, Hugging Face Bucket, that includes all the model checkpoints.  There's an orchestration layer where the rough idea that Claude proposed is like, okay, we've got like a proposer, then agents that would actually like run the experiments.  But before they run experiments, there's a gate check that checks, hey, like, do we, have we pre-registered what the experiments are, what the results we expect to be, and what we'll conclude based on what the end up, the ultimate result is.  So there's a separate review agent that runs on the proposals to check that those are reasonable before they actually run.  And then after it runs, the experiment agent will write up a proposal, and then additionally, there's like a librarian agent that will try to write, like, another shorter review and try to create some additional proposals out of the experimental results.  And then I've got some, like, alerts set. So that the agents can like ping me on my phone when they have questions or when they've like finished various runs.  How are the agents sharing code state?

19:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Each experiment is a new branch off of the NDRL repo.

19:20 - Dan Pandori
  But then after the experiment is run, it's merged into like sort of I've got like just a Dan branch that stores everything.

19:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So like all the agents in a given team, so orchestrator plus the four sub agents that you have are the same CPU?  Oh, no, sorry.

19:43 - Dan Pandori
  There are, what's it called? That's why there are multiple VPS so that we can have multiple agents that don't have to share the same CPU.  But then how do they sync code?

19:56 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  They like push to GitHub or how does it work? Yes.

19:59 - Dan Pandori
  See you next And that's what also is, like, the run pods will actually, you know, get the GitHub to see what they have to actually run.  Okay, and when you run it, you haven't run into issues of, like, the agents are struggling to, like, sync using GitHub, it works fine?

20:20 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  No, I mean, I had to, you know, share some, you know, API keys.

20:25 - Dan Pandori
  I should say, like, that's why, like, the security setup for this is , like, completely. Okay, yeah, we'll put that to the side for a second.  But yeah, I mean, I did notice there was some hiccups, like, one was that, like, oh, when, like, the librarian would finish, there was a log that looked to Claude like I was out of quota, but it actually was just, like, it terminated successfully.  So, you know, there were some hiccups like that that needed to be iterated on so that the agents could distinguish between Hey, we're actually out of quota.  And maybe that's worth getting like, you know, some more cloud quota versus this just being like a normal and expected thing.  Yeah. Can you do something like parser K with this?

21:11 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Like, can I launch like four versions of like four instances of this same pipeline and have them try to have an overseer agent basically say, okay, who did best?

21:25 - Dan Pandori
  Yeah, I have already used it to like, I've like replicated a couple times, like similar experiments, but I haven't done directly like trying to do PASA-K for a given experiment to see, you know, different runs of the same experiment put together by slightly different agents work better.  But yeah, I mean, what I would say is it's all pretty flexible at this point. And so like, you can just kind of tell Claude to if you want some to have something like that, and I expect it would work.  Probably be good to just post the link in the thread for this meetings chat.

21:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I already have.

21:58 - Dan Pandori
  But yeah, I'll share it again. again. Thank Bye.

22:02 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Just so other people can see if they like that, they want to develop their own, et cetera.

22:07 - Dan Pandori
  Totally, yeah. me grab it. And again, I should emphasize that I am ashamed of this and it is not secure, et cetera, et cetera.

22:21 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  My understanding based on how previous projects have gone is that the only people who are going to read the details of it is Claude.

22:29 - Dan Pandori
  So yeah, the humans will look at like, did my number go up using this or not?

22:33 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah. Yeah, that's, yep. Cool. Anyone else want to share like either thoughts on their auto research stuff, stuff they're excited to try or stuff that they have tried?  I was actually, had a quick question.

22:55 - Leon Lenk
  Just, you seem to be a little bit worried about the, like, Communication between agents is, like, do you find in general that, like, the native, like, quad-to-quad communication they build into quad-code is, like, insufficient for, like, purposes like this?  No, I think it's good.

23:13 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think, like, my experience has typically been that when I run everything on a single CPU, so if I have, you know, an agent team on a single CPU executing some auto-research pipeline, I have no issues.  My issue tends to be if I have multiple CPUs running multiple instances of agents that I want to be different, then when one agent team tries to sync the changes that it's made to GitHub with the other agent team, they get pretty confused.  And so, yeah, I found it hard to scale, like, the number of instances that I'm running in parallel if they're all, like, doing ideas that are distinct but all trying to optimize the same benchmark.  For me, that's historically been quite difficult, but I haven't updated the thing that I do for quite a while.

23:58 - Dan Pandori
  Mm-hmm. Mm And out of curiosity, did you have a separate orchestration layer of agents, or was it...? No. I think the hope was that there was no...

24:10 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So at the team level you have an orchestrator, but there wasn't an orchestrator for the teams of teams.

24:18 - Dan Pandori
  So yeah, maybe that would be good to try. Also Dan, what models are you using?

24:23 - Leon Lenk
  Are you only using Opus or Fable or what?

24:26 - Robert Thompson
  I think I'm using Fable, but honestly I'm mostly letting Claude make those decisions.

24:32 - Dan Pandori
  I don't run into the VPS. But yeah, when I'm interacting with it directly, I'm using Fable.

24:40 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I think this is another one of those hyperparameters that I don't know if I've ever measured well, which is if the sub-agents are Opus 4.8 versus Opus 5 versus Fable, what is the relative effectiveness of the pipeline?  It's also part of the reason why I think this parallel instances thing is interesting because... This parser-k thing is interesting because I'm like, oh, well, if I just do one version with, like, Opus is the implementation layer, I do one version where Fable is the implementation layer, and I see that the benchmark difference is, like, negligible, then I'm like, okay, I'll just save myself some money.  Yeah, no one's using chat at all.

25:17 - Robert Thompson
  Oh, you mean Fable? Codex, yeah.

25:20 - Dan Pandori
  Codex.

25:21 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I do think Fable scores higher on coding benchmarks than chat for what it's worth. Higher than Astro? Yeah, Astro generally scores higher than Fable, but I'm pretty sure on coding specifically, Fable is still better.

25:41 - Dan Pandori
  I'm too morally conflicted to use Astro, so I use Quad. Yeah, I mostly am using Astro at the moment.

25:52 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I think it's just, to be honest, I think Codex is a lot worse than Clawed Code, and it always, like, stuns me that that's the case, but.  Yeah. One thing that's worth noting as well is that I don't do a lot of this, but other people do.  Typically you get a lot of uplift from review for doing multi-model stuff, even if it's like a much less capable model that's not the model family that you're using to run the experiments that's doing the review.  So there's a plugin for calling Codex in Clawed, which is quite useful for this. Like obviously the better thing is like you have a version of your pipeline that allows multi-model stuff, but I think a much easier first pass is just use the built-in Clawed tool to call Codex as the review tool.

26:53 - Dan Pandori
  And sorry, you're saying that you think that it's differentially better to have a separate review coming from a separate...  As opposed to the same model family? Yeah. Interesting.

27:04 - Leon Lenk
  Even at lower capability for the review model. So when you're recommending that we get the 200 plan, should we instead get one 100 plan from each provider?

27:20 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I would say get the 200 plan for one provider for Claude or Codex, whichever you prefer. If you need to do some like multi-model stuff, I expect that's going to be not a huge amount.  And so we can probably just use API credits. If we're doing a lot of multi-model stuff, then yeah, it would make sense to do what you're doing.  But I don't think we're at that stage. Does that sound reasonable? Yeah, that makes sense.

27:58 - Leon Lenk
  Yeah.

27:59 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Just start with it. $200 plan, see how far it takes you. If you are running into constraints, then we can change things.

28:07 - Leon Lenk
  Okay, also, just a random question, if I'm missing some context, I didn't do the take-home. Is that where people are getting the numbers about it generalizes beyond a length of six units?  Yes. And I guess I did have my agents try to do the take-home and out of curiosity.

28:32 - Dan Pandori
  I guess it got up to about, I think it got one length of 11 was the best it could do in the runs.  Yeah, I think that's about right.

28:42 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  The length, the stuff which goes beyond length seven should be reviewed because it's typically a very trivial instance. The far more interesting question is how many of the held-out set of textbook problems are the models able to do?  I think when I... When was doing this, I could get one textbook problem solved somewhere between length 8 and 10, but typically it was very noisy, was between 1 or 0, and so I just view that as like, let's just call it 0, for just growing the length of normal proofs, I think you can get very far with that.

29:27 - Robert Thompson
  Sorry, and to be clear, the auto-researched pipeline we're setting up is like identical to the take-home, same setup.

29:36 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I'll talk more about that in a second. Okay. Did someone else have a question? No, terrific, okay. Yeah, let me talk more about that.  So, I think there are a bunch of moving parts to what we could investigate. So maybe there's a good division of labor here where each person might raise their hand to volunteer for each part of it.  So I think the first and always the most important part of any of these pipelines is the initial data set of proofs that we're curating.  So there is one workflow here which is like fix the RL to whatever is like a standard baseline. think the Olmo paper is like a good start on that.  Do people know what I'm talking about when I say like the Olmo RL overview paper? I think you're muted.  I think we did talk about this during the reading group yesterday.

30:56 - Leon Lenk
  I know what Olmo is but I don't know what I haven't read their RL paper.

30:59 - Robert Thompson
  Okay let me...

31:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Let me just post it just in case, I'm just laughing because somebody put out a paper called Two Olmo Two Furious, I think it's this paper, but I'll double check and I'll amend if not.  Okay, so, you know, a lever one is the dataset. So, at some point, we will probably have a package that collates infrastructure for all the different things that we're doing.  Probably the most important part of that package will be whatever functions we're using to programmatically construct the dataset of natural reduction things that we're doing.  And one research question is, given that you fix all the upstream part of the pipelines, how much relative lift can you get from changing what goes into the basic dataset, and what are the parameters under which you should change that?  I.e., what is a good way of having diverse proofs in the constructor of the initial natural deduction proofs? Should we have things like minimal-length proofs, or should we have things like some subset of proofs curated from an interesting dataset of proofs, et cetera, et cetera?  So, I think there is one... So version of one workflow, which is how should we construct the dataset that we're using in order to do both the pre-training and the validation part of the model?  So I think that's one workflow. Another workflow is what about the pre-training process? So given that we fixed the dataset, what should we do in order to make sure that pre-training is doing what we expect it to do?  And what is the relative relationship between pre-training and post-training? So here the most obvious question is how long should you pre-train for and what hyperparameters should you use in your pre-training such that you get an improvement in the RL performance afterwards?  And what is the relationship between the two of those? I would hope that the relationship between the two of those is essentially null.  I would hope that the relationship between... The data set and the RL is very important. The particular pre-training procedure, it would be a little bit sad if those two things depend on each other sensitively, but I don't think I have a good take on this without running the experiments.  So I think that's another work stream that people could have. Quick question there because maybe I'm confused. It seems to me like at least we'll have some trivial relationships between the two.

34:28 - Dan Pandori
  For instance, like if we pre-train and we, you know, we just actually don't run the model to complete, you know, the training process to completion, the RL is going to go worse.  Yeah, totally.

34:40 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  mean, like this is like the more precise way of saying what I what I should have said is within a reasonable tolerance of what achieving low loss on the pre-training data set means, I would hope that the particulars of how you set your learning rate, how you have your learning rate.

35:00 - Dan Pandori
  The scheduler, some of which is very important for getting to a given level of low loss, isn't super important for how RL performs after that.

35:09 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  But I could be wrong, and I don't have a good, like, off-the-cuff, this paper measured this and here is what they found to recommend.  that sound simple? Cool. Then, workstream number three is what RL pipeline should we use for a given fixed dataset and pre-training procedure to get optimal results.  This is basically what the take-home was doing, right? I think that there are, within this, maybe two forward workflows.  One is what is the best off-the-shelf thing we can take from something like Tinker and use in order to be able to just plug in, hopefully, something vaguely close to state-of-the-art.  ... ... ... ... Thank It may be that that's too optimized for large models and it doesn't really do the thing that we wanted to do, but somebody should investigate that.  And the second part of it is, if we were to build something ourselves using reasonable things like PPO or Garpo, what is the relative performance of that and how much work is it to get a pipeline of that working, where unlike in the take home, we do actually understand at least at a rough level what's going on rather than just like let the agents run wild.  So I think that's a decent partition of like four workflows for trying to prove the initial concept of the current natural deduction setting as a model organism for capability emergence.  And, you know, I think one very obvious thing. The mode of this project is we just can't show capability emergence in a compelling way, at which point we just don't have the raw material to make any of the more interesting measurements that we want to do.  So I think this is a good like version zero, which is just like, okay, there are still a bunch of like higher order questions, which if I have time, I'll talk about what natural production tells you that other things don't, why I would be impressed by results here rather than the language model setting and so on.  But as a base thing, if we can't demonstrate good capability emergence in a given subsystem, then we can't do any of those interesting things.  So I think that should be our goal for week one. people should split themselves amongst those four workflows and see to what extent can we demonstrate...  And then maybe once people have had look at it, we can come back and say, okay, where should we focus relative effort for getting higher uplift?

38:11 - Dan Pandori
  Quick question for demonstrating capability emergence. At this point, are we operating from the definition that like capability emergence equals training on like true length longer or like, sorry, successfully generating true length longer proofs than we're in the data set?  Or are we using a different definition?

38:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, so maybe it might even be good to have someone focus on this question specifically. So I think the rough answer is, yeah, I think that's reasonable.  I think a more interesting answer is there are, okay, I think a more interesting answer is, we can make an analogy to, let's say,  The way looks like is there is a period of expert imitation where you train Go on games by people who are good, and then you do RLs.  This is not alpha zero, this is just vanilla alpha Go. In our setting, one of the things that's nice about natural deduction is we can actually draw that line fairly cleanly.  So we can use a set of generators of natural deduction proofs. So this is a series of sub-rules for generating proofs, such that when we combine all of the different generators in a particular set together, we can get all possible natural deduction proofs.  But we would only do the pre-training on a subset of those, what are known as experts in the expert hierarchy.  That I like because it's a setting in which we can explicitly define the boundary between this does what an expert in the pre-training data would do versus this learns a new thing that you would need a new expert to prove.  And so we can formally say whether a given proof falls in one category or the other. So I would be keen to like flesh out this idea a little bit more.  can like send my notes on this. So, yeah, I think there is a I kind of view this as living in the data set camp of like what is the best way to formalize capability emergence?  The most naive way is just do it through proof length. And we should always like at least make that measurement.  But the more interesting way maybe to do something like split the set of natural production proofs into things derivable from a given subfamily of generators.  And then ask for those generators not in the set. How do we transition over? Go ataque Sorry, two questions on this.

41:03 - Robert Thompson
  One, said, like, expert hierarchy, that means just, like, some generator function or, like, some way of generating a proof that the model analysis is before.  And then second, sorry, have you found a way to, like, generate every single possible proof of length X? Yeah, yeah, so we can just do that.  Oh, sorry, sorry.

41:21 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Let me, let me, okay, let me take this too. What was the first part of your question? Just, like, clarifying the definition of expert hierarchy.

41:29 - Robert Thompson
  Yeah, yeah, yeah.

41:30 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So the expert hierarchy basically means there is a function which takes you from a given proof state to the next proof state.  So you can define a series of functions which probably cover the whole natural deduction dataset. And then you can just split them half and half and say, okay, everything can be derived by this family.  And this family means you should imagine a tree where it's like... I apply this rule, then I apply this rule, then I apply this rule in the generator, and then the rest is like, okay, now I have different generators which I can use as nodes of the tree.  So you can prove, or clever people in formal logic have proved, that you can do this for all theorems in propositional logic, and I think we should just steal those proofs.  But still, the specific choice of generators is arbitrary. So, there's not like one universal partition which makes sense to do in natural deduction, but there are choices which you can at least prove will cover the whole set.  As for, do I think that they give us an effective way of generating every instance in the pre-training data efficiently to form the training corpus?  No. Like, I think the, like, we cannot... Obviously, like, parallelize and implement the expert hierarchy to generate different forms of proof, but I think that's no more efficient than just doing the brute force thing of, like, given a premise, append all update rules, choose the ones that you want to then continue to append to, and so on, and you still get exponential blow up of proof size.  I think at the least, this is like an important question for whoever is doing the dataset track of stuff, like, at the least, you should take advantage of the theorems in formal logic that allow you to come up with normal forms for proofs, i.e., two proofs are basically the same thing up to, like, very trivial isomorphisms, but that allows you to, I think, go from, like, exponential growth in the proof size to polynomial, so it's still, like, we want to take advantage of that, at least.  But, no, I don't have a, like, generator function that I would be, like... But yes, use this and that is how you generate your dataset.  I think whoever is doing the dataset side of things should do that. Any other questions about them? None. Okay, good.  Okay, I think it would be good if people either assign themselves to one of those four tracks or propose a different thing that they think it would be more high value for them to work on this week.  And can you just remind me what the four tracks are? Dataset, pre-training, existing RL pipeline, custom RL pipeline. Actually, sorry, I do have one more question, which is like all these imply, like each four tracks imply like two other things are fixed.

44:54 - Robert Thompson
  Yeah. And so like, do you have in mind, like, should we all agree on like what's fixed? And if so, how?  For example, we could say we're all training models of X-parameter size, or by default use this for training dataset.  Yeah, that is a good point.

45:13 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think what I would say is each person should do whatever they think is reasonable for the things that they're going to hold fixed, and then focus on the thing that they're varying.  And then after we've had a week where each person thinks about their part of the pipeline, we can then standardize to whatever each person thought was best in their part.  Okay. But presumably like anchor to what we had in the take-home, so like in terms of parameter size and stuff like that.

45:54 - Robert Thompson
  Anchor roughly, not strongly. Roughly, yeah. I mean, I think you should just use whatever your best discretion is.

46:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I just think the important thing is that make a reasonable choice, but don't spend most of your time on it.  So for example, there is a very reasonable question, which is just like, what should the size of the model be such that we get good performance?  I expect it doesn't matter hugely, but it's probably fairly cheap to say, okay, let's sweep over a size 6 GPU-style transformer and a 12-layer GPU-style transformer.  And, you know, I would do like the cheapest A-B test that sees is this a reasonable choice or not, but I think the defaults in the take-home are probably pretty reasonable.  Like I think your default should be like whatever is in the take-home or whatever you can find in the literature.  Obviously, there's also this other thing of like, if you train a model, if you take a model checkpoint post pre-training...  ... And you're like, oh, , this has really low loss. We can't use that, obviously, right?

47:09 - Dan Pandori
  As in you're saying if it can already generate arbitrary length proofs successfully? No, no, no.

47:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I'm saying if you take a length, if you take a six-block, 500-million-parameter GPT-2-style model, and you evaluate on the training data, and you train on the training data, the pre-training data, and you find that you can't get below 95% accuracy on the validation set for that data set, like, when I say accuracy, I mean like next-token-prediction accuracy.  Obviously, we can't use that model, right?

47:45 - Dan Pandori
  Sorry, you're saying as in if the model just can't, if the model can't learn the data set well, it's not worth using?  The pre-training data set.

47:54 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, yeah. Yeah, like a precondition for all of this is that we can get low loss on the pre-training data set.  I think using the parameters I have in the take-home, I get 95% proof accuracy from predicting next token. So I think that's fine.  But yeah, obviously that is just a basic check that we should... whichever part of the pipeline you're in, you should do.

48:26 - Dan Pandori
  Yeah, I guess to throw my hat in the ring, I'm kind of excited by the dataset generation one of like, you know, testing some of the different datasets generation methods and which ones lead to the best pre-training performance.  Huge. I will put you on that.

48:47 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I'm kind of interested in the... Oh, you can go ahead.

48:50 - Leon Lenk
  No, you got it. Well, was interested in the fourth project of doing RL with like, unique, like the unique RLs that have.  Sorry if that was what we were trying to get, Robbie.

49:05 - Robert Thompson
  No, you're good. I would just say, Dan, I hope you know some math. The data one sounds like the hardest one.  I would probably weakly prefer to do pretraining over the off-the-shelf RL pipeline, but I don't feel that strongly. So, Charles, let me know if you have preferences.  Yeah, I don't think I have a strong preference.

49:24 - Charles Yu
  I think both would be nice for me to do. I'm, like, down to take the other one. Okay, I'll take the pretrain then.  So, I think both for Charles and Leon, one thing...

49:38 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, maybe Leon, if you haven't done Arena before, you might want to, like, the, like, recap the Arena lectures on RL.  I think it's probably, like, the quickest time to insight resource there is. Irrespective of that, I think, for both...  One thing about just any RL pipeline is that our theoretical understanding of it is pretty shallow, and so what that means is that in practice, what people find most value in improving their RL pipelines is having good ways of monitoring it and having heuristics that let you say, Ash, crap, the model stopped learning here, that's why, you know, my actor-critic network wasn't big enough, or my decay rate on this wasn't big enough, so yeah, it's very complicated to know why an RL run is failing, and I think that a crucial piece of infrastructure for everyone going forward is to just have a decent set of dashboards that we monitor to try to see what's happening with our RL runs, so that it's less of a black box, because it is extremely complicated, and, you  A sad version of this project is like we waste a lot of time wondering is it an issue with our RL parameters or is it an issue with something else and unfortunately there's no like MUP type scaling as there is for pre-training which is a reasonable starting point for RL and so it's all it's much much more important to have good dashboards throughout training for those RL runs and so you may find that like investing in that infrastructure is relatively high value when you're doing this.  Sounds good, yeah. Along with that you may find that some of the infrastructure that vendors provide is also high value like I think there is a Xerif version of a monitoring dashboard that Tinker provides and some of it is like very specialized things that don't really that are not really that useful but you should review like whatever people

52:03 - Leon Lenk
  And just to be clear on what the end goal of this is, it's just to see how far we can get RL to improve the length of sequences it can do over pre-training?

52:19 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So I think I would encourage you to think about what you think might be a better measure. So one approach, some of this is dataset dependent, right?  So one thing that Dan will think about is if you think about this generator perspective, does that give you insight into how the dataset can be constructed?  But I think as a minimum, it would be interesting to assess what the generalization is on held out proofs that we know in advance are interesting.  So for me, I think it should be pushed. The repo historically, how I've been doing this is just textbook problems that I've scraped, and I help hold out at least 50% of those at the end of all my custom training as a test set, and that's ultimately what I consider to be the best indicator of whether the model is doing something novel at high length or not.  It's very hard. I've never got the models to do more than 3 out of 50 problems beyond lengths 7 or 8, so that's a very high baseline, and you may want to scrape some other textbook problems or some other well-curated problems or have the models construct some interesting problems for you that you use for that secondary evaluation.  Does that answer your question? question? Yeah, that makes sense.

54:01 - Leon Lenk
  And just to be totally clear, like the ndrl repo already includes all the stuff that was in the take-home, right?  It might do, but I can check.

54:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, there is like a bunch of experiments that I need to push to it, which got lost in another cluster, which are basically like versions of solutions to the take-home.  Cool. Other questions, concerns, supports, queries. Actually, have one more question about mine as well for pre-training.

54:44 - Robert Thompson
  I presume my goal is to optimize pre-training such that on some fixed RL pipeline afterwards, I get the most proof and not optimize pre-training such that I get to the lowest loss the fastest.

54:53 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  That's right. That's right.

54:53 - Robert Thompson
  That's right.

54:54 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I expect those two things to be related, but I think an interesting finding is like... Perhaps you don't need to like get the lowest lot.  mean, I mean, one just straightforwardly interesting question is like how much of a difference does it make, right? So it'd be very interesting if we find that, like what I would expect to happen is it's fairly wide band away from like the lowest loss you can get on a given pipeline such that the RL results are unchanged.  It'd be very interesting if you can pre-train too much for your RL pipeline to generalize effectively. I would be very surprised if we find that, but it would be interesting.  And there's a bunch of questions about like, are there different pre-training procedures that more or less incentivize RL and so on?  And like for all of these things, right, like the workflow is something like you think about it, you see what does the literature say, you then say, okay, of the things I've thought about and what the literature says, here are the things that are plausible to try, let me try them.

55:58 - Robert Thompson
  And here we're kind of like in...

56:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So if somebody has already done something in the literature, that's fantastic. Let's just use state-of-the-art. Makes sense. Also, sorry, is tokenizer and stuff like that in scope for me?  I viewed that as more of a dataset question, but you should feel free to experiment with it if you want.  I think one question that I'm kind of interested in is we can use a tokenization of natural deduction, which is basically lean.  So we can put natural deduction proofs into lean in a straightforward way because at the end of the day, they're just formal logic statements.  And it's very, very close to the tokenization you would do if you were just doing it as you would in a textbook.  I think at least we should have a converter between whatever tokenization we have and the lean tokenization. But I'd be very curious if like...  Yeah, I expect that if we do it in Lean, then if we ask other models to review, like Novelty or whatever approved, it'll be much easier for them to do.  I'd be curious if it just gives us a lift in the pre-training to do that or not. I I expect it doesn't, but it might be interesting to try.

57:22 - Dan Pandori
  Yeah, I did seek Cloud on this, and it found that Lean did better than the prior tokenization scheme. This is for a model doing proofs from some in-context examples, or this is for pre-training?  My understanding is for pre-training, but I won't lie and tell you that I looked at this experiment that carefully.  I can give you a better answer tomorrow. That's interesting.

57:53 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I would have expected that in pre-training it would make no difference, but it could. but you season. can can  If it does, I think that's kind of interesting. Yeah, maybe something to... Maybe you can just post, like, the summary.md that Claude came up with in the Sunday meeting, and then either you or Robbie or anyone else interested can have a look at it.  Can do. Cool. Any other questions? Yeah, just for the architecture that you picked, I haven't looked at it.

58:28 - Leon Lenk
  So, like, are we going to try iterating on that to see if it's, like, the Soda architecture? Because, like, I know it's, like, changes quite a bit.  Like, for example, I just started using loop transformers. Architecture is in, like, GPT-2? I mean, like, the choice of, like, using, like, rope versus, like, cosine, sine embeddings, for example.  Yeah, for me, those are pre-training questions.

59:05 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I expect that for natural deduction, it won't make a ton of difference. I think there is a very interesting research project, which is if we allow ourselves a semi-looped architecture, like, for example, the hyper-manifold connections that DeepSeq used in V4, V3, I think it was V4, which is like plus epsilon recursion to standard transformer architecture.  It's a really interesting question to ask, like, does that differ, how does that affect RL performance? I expect, and I think this is a good thing to read, both the DeepSeq V4.1 flash paper and the Olmo, you know, modern RL training overview paper for, I expect that the vast majority of the, like, robust difference...  should come from improvements in the pre-training data, but I don't know. Pressure's on, Dan.

1:00:12 - Robert Thompson
  Yeah, mean, differentially, like, the data set is pretty, pretty important.

1:00:18 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  So there was something I wanted to say from that question. Any other questions, concerns, comments, objections, claims, conjectures? None.  Terrific. Okay. All right. If you have any preference for who you want to be paired with because of time difference or other issues, please let me know.  Otherwise, I'll... Just pair you randomly into workstreams of two for the midweek meeting. Wait, sorry, what is the point of the midweek meeting?  Yeah, so the midweek meeting is like the three of us, you, your workstream partner and me, will try and like co-work on whatever it is that you're working on that week to see can I help you make progress on the low level of what you're doing.  As opposed to the weekly meeting on the Sundays where we basically review results and if we can unblock certain issues then we should certainly try, but mostly we're just like trying to get clarity on future direction.

1:01:46 - Leon Lenk
  Okay, that makes sense, thanks. Cool, any other questions, concerns, comments?

1:01:57 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Terrific. All right folks, actually wait, sorry. Is this going to be the recurring meeting time for this group meeting, or are we back to Saturday morning?

1:02:05 - Dan Pandori
  We're back to Saturday morning, unless anyone tells me that that is problematic for them.

1:02:13 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Speak now, forever hold your peace. Good. Okay, terrific. Yeah, this was a one-off thing because someone messaged me to say, like, oh, should we not do it this week?  Cool. Yeah, I think that's all I had. One thing, as you're working on this project, I think it's very helpful to keep all of your thinking within the repo.  And so for me, the way I do that is most of the stuff that I do, if I write a tech document or a Google Doc, I then convert it into .md files, and then I put it into the documentation side of the repo.  I would encourage you all to do the same, because it's helpful for other people when they're trying to understand what you did.  It's also very helpful for the agent. to understand what you did and build up your ideas and it's also helpful for people who are managing agents to tell them to do the things that they're excited to do from what you were saying so yeah as far as you can your principle should be that to keep everything in the repo so that it can be accessed by everybody else alright folks I will see you at some point in the middle of the week cheers