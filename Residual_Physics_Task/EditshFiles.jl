using Pkg; Pkg.activate(".")

#script to edit the contents of .sh files in the current directory
#such as this one:
# "#!/bin/bash
# #SBATCH --partition=short
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --time=5:30:00
# #SBATCH --mem-per-cpu=12000M
# #SBATCH --account=cosc016682
# #SBATCH --array=1-5
# #SBATCH --job-name=SpectralRadiusLow_Hybrid_ASF
# #SBATCH --output=SpectralRadiusLow_Hybrid_ASF_-%a.o

# cd ${SLURM_SUBMIT_DIR}
# module load languages/julia/1.10.3
# echo JOB ID: ${SLURM_JOBID}

# echo SLURM ARRAY ID: ${SLURM_ARRAY_TASK_ID}

# echo Working Directory: $(pwd)

# echo Start Time: $(date)

# julia Task2_ParameterSweep.jl ${SLURM_ARRAY_TASK_ID} MeanDegree 5 /user/home/as15635/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Settings_and_GroundTruth /user/work/as15635/output_data/ExtKuramoto/ Hybrid

# echo Finish Time: $(date)

# "

pwd()
cd("Residual_Physics_Task")
# Get all .sh files in the current directory
sh_files = readdir(pwd(), join=true) .|> x -> endswith(x, ".sh") ? x : nothing
sh_files = filter(x -> x != nothing, sh_files)

# Loop through each .sh file
for file in sh_files
    # Read the contents of the file
    content = read(file, String)

    # # change array range to 1-10
    # content = replace(content, r"#SBATCH --array=\d+-\d+" => "#SBATCH --array=1-10")
    # #change the "SpectralRadiusLow" part of the job name and output to "MeanDegree"
    # content = replace(content, r"SpectralRadiusLow" => "MeanDegree")
    # #Change cpus-per-task to 4
    # content = replace(content, r"#SBATCH --cpus-per-task=\d+" => "#SBATCH --cpus-per-task=4")
    # #after "julia" add "--threads 4 "
    # content = replace(content, r"julia " => "julia --threads 4 ")

    #change MeanDegree to KnowledgeRatio
    # content = replace(content, r"MeanDegree" => "KnowledgeRatio")
    #if time line is 3:59:00
    # change it to 00:30:00
    content = replace(content, r"#SBATCH --time=0:59:59" => "#SBATCH --time=00:20:00")

    # Write the modified content back to the file
    open(file, "w") do io
        write(io, content)
    end
    println("Updated $file")
end

#change the filenames of the files so MeanDegree becomes KnowledgeRatio
for file in sh_files
    # Get the filename without the path
    filename = basename(file)

    # Replace "MeanDegree" with "KnowledgeRatio" in the filename
    new_filename = replace(filename, r"MeanDegree" => "KnowledgeRatio")

    # Rename the file
    mv(file, joinpath(dirname(file), new_filename))
    println("Renamed $filename to $new_filename")
end


# load all ".o" files in the current directory

o_files= readdir(pwd(), join=true) .|> x -> endswith(x, ".o") ? x : nothing
o_files = filter(x -> x != nothing, o_files)
#find all files with "CANCELLED" in their contents and list them:
cancelled_files = []
for file in o_files
    # Read the contents of the file
    content = read(file, String)

    # Check if "CANCELLED" is in the content
    if occursin("CANCELLED", content)
        push!(cancelled_files, file)
    end
end
println("The following files have 'CANCELLED' in their contents:")
for file in cancelled_files
    println(file)
end