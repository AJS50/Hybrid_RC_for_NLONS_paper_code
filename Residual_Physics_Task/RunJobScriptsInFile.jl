cd(@__DIR__)

#collect all .sh file names
files=readdir() 
job_scripts=files[findall(x->occursin(".sh",x),files)]

#if you want to exclude anything or filter the scripts further than just "contains .sh")
# job_scripts=job_scripts[findall(x->occursin("MeanDegree",x),job_scripts)]

#run sbatch command for each.
for job in job_scripts
    # println(job) #use this and comment out the command if you want to just see what it will run before running it. 
    run(Cmd(["sbatch", job]))
    sleep(2) #may not be neccessary, but added to try and avoid clashes in precompilation
end

