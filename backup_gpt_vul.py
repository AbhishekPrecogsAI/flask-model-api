
def run_detection_with_context():
    # Path to the codebase folder
    folder_path = "./data/"

    # Step 1: Chunk the source files
    print("Chunking source files...")
    chunks, chunk_mapping = chunk_source_files(folder_path)

    # Step 2: Index the chunks
    print("Indexing chunks...")
    index, embeddings = index_chunks(chunks)

    # index, embeddings, tokenizer,  model = index_chunks(chunks)

    # Step 3: user input code

    code_snippet2 = """static int perf_trace_event_perm(struct ftrace_event_call *tp_event,
    				 struct perf_event *p_event)
             {
                /* The ftrace function trace is allowed only for root. */
                if (ftrace_event_is_function(tp_event) &&
                    perf_paranoid_kernel() && !capable(CAP_SYS_ADMIN))
                    return -EPERM;

                /* No tracing, just counting, so no obvious leak */
                if (!(p_event->attr.sample_type & PERF_SAMPLE_RAW))
                    return 0;

                /* Some events are ok to be traced by non-root users... */
                if (p_event->attach_state == PERF_ATTACH_TASK) {
                    if (tp_event->flags & TRACE_EVENT_FL_CAP_ANY)
                        return 0;
                }

                /*
                 * ...otherwise raw tracepoint data can be a severe data leak,
                 * only allow root to have these.
                 */
                if (perf_paranoid_tracepoint_raw() && !capable(CAP_SYS_ADMIN))
                    return -EPERM;

                return 0;
            }
        """

    # Step 3: Retrieve relevant chunks
    print("Retrieving relevant chunks...")
    # relevant_chunks = retrieve_relevant_chunks(code_snippet2, chunks, index, tokenizer, model, top_k=3)
    relevant_chunks = retrieve_relevant_chunks(code_snippet2, chunks, index, top_k=3)

    logger.info("Starting vulnerability analysis...")
    result = analyze_code_vulnerability_with_context(code_snippet2, relevant_chunks)

    if isinstance(result, DetectionResult):
        # print(result.json(indent=4))
        logger.info(json.dumps(result.model_dump(), indent=4))
    else:
        logger.error("Analysis failed with error: %s", result.get("error"))

    logger.info("Showing the diff...")

    # Generate Markdown diff
    relevant_lines = [line.lineNum for line in result.vulnerabilityLines]

    markdown_diff_1 = generate_incident_diff(code_snippet2, result.fixCode, relevant_lines)
    logger.info(markdown_diff_1)

    # Optionally, convert Markdown to HTML for better viewing (e.g., in a browser)
    html_diff = markdown2.markdown(markdown_diff_1)

    file_id = 'test'  # use commit id

    with open(f"./{file_id}_diff.html", "w") as f:  # Save as HTML if needed
        f.write(html_diff)


def analyze_code_vulnerability_with_context(code_snippet: str, retrived_chunks: [str]) -> Union[DetectionResult, dict]:
    """
    Analyze a code snippet for vulnerabilities using OpenAI's API with context retrieved using RAG.

    Args:
        code_snippet (str): The code snippet to analyze. perhaps put at function level
        retrived_chunks: list of code for context

    Returns:
        Union[DetectionResult, dict]: The structured analysis result or an error message.
    """
    try:

        prompt = f"""
        You are an advanced cybersecurity expert proficient in all programming languages. 
        Analyze the following code snippet for vulnerabilities at the function level with context of sourcefile.
        Before providing your final answer, internally reason through the code's property graph—including its Abstract Syntax Tree (AST), 
        Control Flow Graph (CFG), and Program Dependence Graph (PDG)—to identify potential vulnerabilities. 
        Do not output this internal chain-of-thought; only provide the final result in the JSON format specified below.\n\n
        Following the steps for output.
        1. Identify the programming language of the code snippet.
        2. Analyze the code for any vulnerabilities or security issues within the context provided.
        3. If vulnerabilities are found:
           - Specify the type of vulnerability.
           - Identify the vulnerable lines of code with the line numbers and the actual code.
           - Provide a detailed explanation of why these lines are vulnerable and the potential risks.
           - Suggest a complete and efficient fix for the vulnerable code based on root cause and best practise, and **return the entire code block with the fix** included (not just the modified lines).
        4. Format your entire response as valid JSON.

        ### Code snippet:
        {code_snippet}

        ### Source file context:
        {retrived_chunks}
        """

        # Step 4: Call OpenAI API with the constructed prompt
        # client = OpenAI(api_key=API_KEY)
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert."},
                {"role": "user", "content": prompt}
            ],
            response_format=DetectionResult
        )
        analysis_result = response.choices[0].message.parsed
        logger.info("Vulnerability analysis completed successfully. See result below")

        logger.info(analysis_result)

        return analysis_result

    except Exception as e:
        logger.error(f"Error during vulnerability analysis: {str(e)}")
        return {"error": str(e)}
