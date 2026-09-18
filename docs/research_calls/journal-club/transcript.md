Automated Alignment Researcher discussion - September 17
VIEW RECORDING - 6 mins (No highlights): https://fathom.video/share/sxJs3D6YXyoaMZQVRW-1faxHShxE4Jnr

---

0:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  What did you do?

0:06 - Dan Pandori
  Cool.

0:07 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I should probably run off in a bit. I just want to say one quick thing, if that's okay. Please.  So I think one thing, I'll say a bit more about this on Sunday, but one thing that would encourage all of you to think about is if you're considering doing safety research, which I hope you are, the vast majority of AI safety research now gets mediated through the agents.  And so having good agentic pipelines on different timescales that you apply is a fundamental research skill. And insofar as the hope for me to some extent is to give you all good tools and skills for doing AI safety work, I would encourage you to think about, like, both.  The paper that we're talking about here and the other auto research things as part of your research infrastructure which you use to do AI safety work and see this project as to some extent a testbed for how well you use that infrastructure.  So I would assume that many of you are considering applying to things like Anthropic Fellows. One of the great things about Anthropic Fellows is that you get unlimited clause.  And so, you know, I know people who have used like more than a million dollars worth of credits in Anthropic Fellows.  And one thing you should bear in mind is that like, OK, if you want to do Anthropic Fellows, you know, one thing that will be incredibly useful to you is to be able to absorb something like a million dollars worth of credits of Claude.  For that, you should be able to build really good research pipelines. And I would encourage you to have a motivation of this project to just get really good at that skill.  So part of my interest in like having you read this paper and also the broader auto research literature is just building that for yourself.  So. So. So.

2:00 - Robert Thompson
  Anyone have a good dummy auto-research, like I confess I've never done this properly, I feel like I've only shipped, I've worked a lot with the agents, but like on prod code where I don't do auto-research, anyone have a good dummy task that's like fun to get a rep at this, or should I just like directly work on this project and that's mine?

2:14 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Yeah, I would say like RL-ing a model to get longer proofs, I mean you can do, I mean you can like a good meta benchmark, right, a benchmark of your auto-research thing is like just do the nano-GPT-Karpati task, right?  There's a good auto-research, there's a good auto-research repo that did, okay, so another good auto-research task is jailbreak, so have you read the Claudini paper?  No, I have not. Okay, let me send it in the Slack, so this was like a Matt Scholar when like Karpati came out with auto-research, basically was like, let's try this for jailbreaks, and it worked really, really well.  So, you know, I'm not sure how much I want to encourage you to just come up with... I'm sure the Chinese labs have been doing this for a long time, so I'm not that bothered about it, but that's a good task on which you can try what a person without a huge amount of infrastructure would come up with for their agent pipeline.

3:18 - Leon Lenk
  One thing I'm a little bit confused about in general with these sort of pipelines is, how do you actually test that your pipeline is better than a counterfactual pipeline?  Yeah, this is a great question.

3:28 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  I have not seen any good infrastructure for tracking, like, you know, A-B testing which parts of your auto-research pipeline are doing work and which of them are not, and this is a huge problem, and I encourage you to explore and come up with a good pipeline for that.

3:45 - Dan Pandori
  I mean, I do think that the paper that this meeting was nominally about did some good ablation studies on that, which was kind of cool.  Yeah, but when you do it yourself, you'll find like, ah, , like, is me, like, sub-delegating to this team of free agents to...

4:00 - Dmitry Manning-Coe (dmanningcoe@gmail.com)
  Do you some review process like actually good or is the review agent just being so adversarial that Claude always collapses to like the difference between seed four and five was six percent and I'm really unhappy about that so let me run another hundred seeds like it's really hard to know I think in those instances whether that change is good or bad and I would love to have a better framework for what Leon is talking about anyway so I should run and stop sabotaging your meeting but good to see you folks and I will see you on Sunday.

4:28 - Dan Pandori
  Good to see you okay so we're substantially over time but I think this was a good discussion of the paper the one thing I'd maybe say is I think that maybe having like recurring like reading group meetings might be helpful if somebody would like to like volunteer to sort of like you know host the next one and like put together some discussion questions I'd love that otherwise I will try to do so.  So I think if this time works for. Most people, I think that's just like the natural shelling point default, let me know if it doesn't, but yeah, and if folks, yeah, so what I'll say is I'm going to plan to, I guess, the team meeting on Sunday, I'll propose like next paper to discuss next Thursday, but if you have any paper that you'd like to do, just message me before then, and I would be delighted to have this be like a rotating thing of who sort of leads it.

5:32 - Robert Thompson
  If I find a paper, I'll absolutely, I'm happy to lead it. Awesome. Yeah. I don't have one on the top of my head, but if you want to just like tentatively tap me to be next Thursday, 4 p.m., I can do it.  All right, cool.

5:45 - Dan Pandori
  I'll, yeah, I'll share a, I'll create another meeting with the same people and give everybody edit access and all that.  Okay, sounds good.

5:53 - Robert Thompson
  All right. Good meeting y'all. Nice to meet y'all. Nice to meet you guys. See you. Yeah.