# Repo design principles

This doc outlines the principles of this repo. The Repo serves several roles:

1. It is a workspace interface for day-day work.
2. It is the database for agents to use as context.
3. It is a store of knowledge for our project.

<!-- Aside: would the ideal workflow use separate tools for this? -->

What properties do these uses imply for the repo?

1. It should support being used for very fast iteration
    - to first order, this is separate branches that don't interact.
2. Agents must be able to find the relevant context without degrading in performance:
    - 

3. It should be easy for a human to navigate:
    - Agents should not pollute the folder with lots of one-off files.
    - Keep separate solid and speculative work.
    
