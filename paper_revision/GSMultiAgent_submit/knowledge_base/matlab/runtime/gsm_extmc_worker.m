%GSM_EXTMC_WORKER Persistent MATLAB worker for external-MC jobs.
%  Polls job_dir/pending/*.req.json, runs eval_str via evalc, writes
%  job_dir/done/<job_id>.res.json.  Keep process alive (no exit here).
function gsm_extmc_worker(job_dir)
if isstring(job_dir); job_dir = char(job_dir); end
pending = fullfile(job_dir, 'pending');
running = fullfile(job_dir, 'running');
done_dir = fullfile(job_dir, 'done');
for d = {pending, running, done_dir}
    if ~exist(d{1}, 'dir'); mkdir(d{1}); end
end
set(0, 'DefaultFigureVisible', 'off');
fprintf('[gsm_extmc_worker] ready dir=%s\n', job_dir);
idle_sleep = 0.15;
while true
    files = dir(fullfile(pending, '*.req.json'));
    if isempty(files)
        pause(idle_sleep);
        continue;
    end
    [~, ord] = sort([files.datenum]);
    name = files(ord(1)).name;
    src = fullfile(pending, name);
    dst = fullfile(running, name);
    try
        movefile(src, dst);
    catch
        continue;
    end
    ok = true;
    out = '';
    job_id = '';
    try
        req = jsondecode(fileread(dst));
        if isfield(req, 'job_id'); job_id = char(req.job_id); end
        if isfield(req, 'eval_str') && ~isempty(req.eval_str)
            out = evalc(char(req.eval_str));
        elseif isfield(req, 'script_path') && ~isempty(req.script_path)
            sp = strrep(char(req.script_path), '''', '''''');
            out = evalc(sprintf('run(''%s'');', sp));
        else
            ok = false;
            out = 'job missing eval_str/script_path';
        end
    catch ME
        ok = false;
        out = getReport(ME, 'extended', 'hyperlinks', 'off');
    end
    if isempty(job_id)
        [~, job_id] = fileparts(name);
        job_id = strrep(job_id, '.req', '');
    end
    res = struct('ok', ok, 'stdout', out, 'job_id', job_id);
    res_path = fullfile(done_dir, [job_id '.res.json']);
    try
        fid = fopen(res_path, 'w');
        if fid < 0
            error('cannot write %s', res_path);
        end
        fwrite(fid, jsonencode(res), 'char');
        fclose(fid);
    catch ME
        fprintf('[gsm_extmc_worker] write failed: %s\n', ME.message);
    end
    try
        delete(dst);
    catch
    end
end
end
