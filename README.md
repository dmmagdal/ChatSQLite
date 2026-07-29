# Chat SQL

This is a proof-of-concept for building out an agent that allows layman users to interact with SQL databases and perform analysis around that data.


### Notes

 - Data
     - The data sourced for this repository comes from sources used in [this repo](https://github.gatech.edu/dmagdaleno3/CSE6242-Project). That project was related to aggregating and presenting FEC 
     - There is a lot of work that should go into data processing when setting this up. This repository is a proof-of-concept to test how LLMs & agents can work to provide a seemless experience to non-technical people. As such, the data currently used is ingested with minimal preprocessing and cleaning. If you are applying this to your own data, be sure to invest the effort in data engineering, as you'll see how that effects the quality of the agent as well as the handling/retrieval of the SQL data.
     - Be sure to provide as much supplemental information or context around the columns as possible. In cases where you have to join data from tables that are generated via different sources, that context will be what allows the LLM to write the necessary `JOIN` commands.
         - Alternatively, you could also make sure that naming conventions across tables align in the first place. This is the better choice of action because it requires less text in the input LLM context and because it aligns with best practices for database administrators and developers. That being said, the former bullet point above will likely be a common circumstance for developers.