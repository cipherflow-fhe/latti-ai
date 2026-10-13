# Attach initialized submodules to their configured branch at the pinned commit.
find_package(Git QUIET)

function(bind_submodule_branch parent module)
    set(module_dir "${parent}/${module}")
    if(NOT GIT_FOUND OR NOT EXISTS "${module_dir}/.git")
        return()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${parent}" config -f .gitmodules --get "submodule.${module}.branch"
        OUTPUT_VARIABLE branch OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0 OR branch STREQUAL "")
        return()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${module_dir}" rev-parse HEAD
        OUTPUT_VARIABLE head OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Cannot determine ${module} commit")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${module_dir}" symbolic-ref --quiet --short HEAD
        OUTPUT_VARIABLE current_branch OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(current_branch STREQUAL branch)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${module_dir}" rev-parse --abbrev-ref --symbolic-full-name "@{upstream}"
            OUTPUT_VARIABLE upstream OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        )
        if(upstream STREQUAL "origin/${branch}")
            message(STATUS "${module}: ${branch} tracks ${upstream} at ${head}")
            return()
        endif()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${module_dir}" rev-parse --verify "refs/heads/${branch}"
        OUTPUT_VARIABLE branch_head OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE result ERROR_QUIET
    )
    if(result EQUAL 0)
        if(NOT branch_head STREQUAL head)
            message(FATAL_ERROR "${module}: existing ${branch} differs from HEAD; reconcile the branch before building")
        endif()
    else()
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${module_dir}" branch --no-track "${branch}" HEAD
            RESULT_VARIABLE result ERROR_VARIABLE error
        )
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "Cannot create ${module} branch: ${error}")
        endif()
    endif()

    foreach(action IN ITEMS checkout branch)
        if(action STREQUAL "checkout")
            set(arguments checkout "${branch}")
        else()
            set(arguments branch "--set-upstream-to=origin/${branch}" "${branch}")
        endif()
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${module_dir}" ${arguments}
            RESULT_VARIABLE result OUTPUT_QUIET ERROR_VARIABLE error
        )
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "Cannot bind ${module} to origin/${branch}: ${error}")
        endif()
    endforeach()
    message(STATUS "${module}: ${branch} tracks origin/${branch} at ${head}")
endfunction()
