from phoena_translator.config import AppConfig
from phoena_translator.task_store import TaskRepository, TaskStore
from phoena_translator.web import WebDependencies, create_app


def test_failed_epub_resume_preserves_task_and_enqueues_original_source(tmp_path):
    uploads, outputs, progress = (tmp_path / name for name in ('uploads', 'outputs', 'progress'))
    for directory in (uploads, outputs, progress):
        directory.mkdir()
    source = uploads / 'book.epub'
    source.write_bytes(b'original book archive')
    config = AppConfig(upload_dir=uploads, output_dir=outputs, progress_dir=progress,
                       log_file=tmp_path / 'log')
    store = TaskStore(progress, uploads, outputs)
    tasks = {'book': {'type': 'epub', 'status': 'failed', 'filename': 'book.epub', 'error': 'validation'}}
    queued = []
    dependencies = WebDependencies(
        tasks=TaskRepository(store, tasks), ensure_runtime=lambda: None,
        enqueue_task=lambda *args: queued.append(args) or True,
    )
    client = create_app(config, dependencies).test_client()
    response = client.post('/api/task/book/resume', json={})
    assert response.status_code == 200
    assert tasks['book']['status'] == 'queued'
    assert tasks['book']['error'] is None
    assert queued[0][:3] == ('book', 'epub', str(source))
    assert client.post('/api/task/book/resume', json={}).status_code == 400


def _client(tmp_path, record):
    uploads, outputs, progress = (tmp_path / name for name in ('uploads', 'outputs', 'progress'))
    for directory in (uploads, outputs, progress):
        directory.mkdir()
    (uploads / 'book.epub').write_bytes(b'original book archive')
    config = AppConfig(upload_dir=uploads, output_dir=outputs, progress_dir=progress,
                       log_file=tmp_path / 'log')
    store = TaskStore(progress, uploads, outputs)
    tasks = {'book': record}
    queued = []
    dependencies = WebDependencies(
        tasks=TaskRepository(store, tasks), ensure_runtime=lambda: None,
        enqueue_task=lambda *args: queued.append(args) or True,
    )
    return create_app(config, dependencies).test_client(), tasks, queued


def test_completed_epub_with_failed_segments_can_retry_them(tmp_path):
    client, tasks, queued = _client(tmp_path, {
        'type': 'epub', 'status': 'completed', 'filename': 'book.epub', 'output': '/x/book.epub',
        'failed_segment_count': 3, 'warning': '3段未通过校验：3段保留原文；可用“重试未通过段落”再试'})
    response = client.post('/api/task/book/resume', json={})
    assert response.status_code == 200
    assert tasks['book']['status'] == 'queued'
    assert tasks['book']['warning'] is None and tasks['book']['output'] is None
    assert queued[0][:2] == ('book', 'epub')


def test_completed_epub_without_failed_segments_cannot_resume(tmp_path):
    client, tasks, queued = _client(tmp_path, {
        'type': 'epub', 'status': 'completed', 'filename': 'book.epub', 'failed_segment_count': 0})
    assert client.post('/api/task/book/resume', json={}).status_code == 400
    assert tasks['book']['status'] == 'completed' and queued == []
