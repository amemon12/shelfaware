import importlib.util
import os
from pathlib import Path
import socket
import stat

import pytest
from dotenv import dotenv_values

path = Path(__file__).resolve().parents[2] / 'scripts' / 'start_backend.py'
spec = importlib.util.spec_from_file_location('backend_only_launcher', path)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_lan_token_generation_preserves_provider_keys_and_is_not_printed(tmp_path,capsys):
    env=tmp_path/'.env'
    env.write_text('PERCEPTRON_API_KEY=private-perceptron\nGEMINI_API_KEY="private-gemini"\nSHELF_API_TOKEN=\n',encoding='utf-8')
    environment={}
    token=launcher.ensure_lan_token(tmp_path,environment)
    assert len(token)>=32 and environment['SHELF_API_TOKEN']==token
    values=dotenv_values(env)
    assert values['PERCEPTRON_API_KEY']=='private-perceptron' and values['GEMINI_API_KEY']=='private-gemini'
    assert values['SHELF_API_TOKEN']==token
    assert launcher.ensure_lan_token(tmp_path,{})==token
    connection=launcher.save_phone_connection(tmp_path,token,[('en0','192.168.7.20')])
    text=connection.read_text(encoding='utf-8')
    assert token in text and 'http://192.168.7.20:8000' in text
    assert 'private-perceptron' not in text and 'private-gemini' not in text
    assert capsys.readouterr().out==''
    if os.name!='nt':
        assert stat.S_IMODE(connection.stat().st_mode)==0o600
        assert stat.S_IMODE(env.stat().st_mode)==0o600
        assert stat.S_IMODE(connection.parent.stat().st_mode)==0o700


def test_existing_file_token_preserved_and_matching_environment_accepted(tmp_path):
    env=tmp_path/'.env'
    env.write_text('SHELF_API_TOKEN="existing-local-token"\nPERCEPTRON_API_KEY=keep-me\n',encoding='utf-8')
    assert launcher.ensure_lan_token(tmp_path,{})=='existing-local-token'
    environment={'SHELF_API_TOKEN':'existing-local-token'}
    assert launcher.ensure_lan_token(tmp_path,environment)=='existing-local-token'
    assert dotenv_values(env)['SHELF_API_TOKEN']=='existing-local-token'
    assert dotenv_values(env)['PERCEPTRON_API_KEY']=='keep-me'


def test_conflicting_environment_token_rejected_without_mutation_or_secret_output(tmp_path,capsys):
    env=tmp_path/'.env'
    original='SHELF_API_TOKEN="stored-token-value"\nGEMINI_API_KEY=private-provider-key\n'
    env.write_text(original,encoding='utf-8')
    environment={'SHELF_API_TOKEN':'environment-token-value'}
    with pytest.raises(RuntimeError,match='differs from .env') as error:
        launcher.ensure_lan_token(tmp_path,environment)
    assert env.read_text(encoding='utf-8')==original
    assert environment=={'SHELF_API_TOKEN':'environment-token-value'}
    assert 'stored-token-value' not in str(error.value)
    assert 'environment-token-value' not in str(error.value)
    assert capsys.readouterr().out==''


def test_malformed_token_rejected_without_printing_it(tmp_path):
    with pytest.raises(RuntimeError) as error:
        launcher.ensure_lan_token(tmp_path,{'SHELF_API_TOKEN':'secret\nheader'})
    assert 'secret' not in str(error.value) and not (tmp_path/'.env').exists()


def test_occupied_loopback_port_refuses_without_killing_server():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as occupied:
        occupied.bind(('127.0.0.1',0))
        occupied.listen(1)
        port=occupied.getsockname()[1]
        with pytest.raises(RuntimeError,match='never kills'):
            launcher.ensure_port_available('127.0.0.1',port)
        assert occupied.getsockname()[1]==port


def test_mac_addresses_are_actual_interface_results(monkeypatch):
    results={
        ('networksetup','-listallhardwareports'):'Hardware Port: Wi-Fi\nDevice: en0\n\nHardware Port: Ethernet\nDevice: en1\n',
        ('ipconfig','getifaddr','en0'):'192.168.4.21\n',
        ('ipconfig','getifaddr','en1'):'',
    }
    monkeypatch.setattr(launcher,'command_output',lambda args:results.get(tuple(args),''))
    assert launcher.lan_candidates('Darwin')==[('en0','192.168.4.21')]


def test_mac_ifconfig_fallback_ignores_loopback(monkeypatch):
    def output(args):
        return 'lo0: flags\n\tinet 127.0.0.1 netmask\nen0: flags\n\tinet 10.0.0.42 netmask\n' if args==['ifconfig'] else ''
    monkeypatch.setattr(launcher,'command_output',output)
    assert launcher.lan_candidates('Darwin')==[('en0','10.0.0.42')]


def test_mac_ipv6_clat_address_is_not_advertised_to_phone(monkeypatch):
    def output(args):
        return 'en0: flags\n\tinet 192.0.0.2 netmask 0xffffffff\n' if args==['ifconfig'] else ''
    monkeypatch.setattr(launcher,'command_output',output)
    assert launcher.lan_candidates('Darwin')==[]


@pytest.mark.parametrize('port',[8000,8001])
def test_loopback_launch_executes_only_uvicorn_and_never_initializes_lan_token(tmp_path,monkeypatch,capsys,port):
    (tmp_path/'backend').mkdir()
    commands=[]
    checks=[]
    monkeypatch.setattr(launcher,'ensure_port_available',lambda host,port:checks.append((host,port)))
    monkeypatch.setattr(launcher,'ensure_lan_token',lambda *args:pytest.fail('Loopback should not initialize LAN credentials'))
    monkeypatch.setattr(launcher.os,'chdir',lambda path:None)
    monkeypatch.setattr(launcher.os,'execv',lambda executable,args:commands.append(args))
    launcher.launch(tmp_path,False,port)
    assert checks==[('127.0.0.1',port)]
    assert len(commands)==1
    assert commands[0][1:]==['-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(port),'--no-access-log']
    output=capsys.readouterr().out
    assert f'http://127.0.0.1:{port}' in output and 'Node' not in output


def test_lan_launch_checks_port_before_writing_credentials(tmp_path,monkeypatch):
    def refuse(host,port):
        assert host=='0.0.0.0' and port==8001
        raise RuntimeError('occupied')
    monkeypatch.setattr(launcher,'ensure_port_available',refuse)
    monkeypatch.setattr(launcher,'ensure_lan_token',lambda *args:pytest.fail('Must refuse before credential changes'))
    with pytest.raises(RuntimeError,match='occupied'):
        launcher.launch(tmp_path,True,8001)
    assert not (tmp_path/'.env').exists()


def test_custom_lan_port_matches_probe_server_connection_file_and_output(tmp_path,monkeypatch,capsys):
    checks=[]
    commands=[]
    monkeypatch.setattr(launcher,'ensure_port_available',lambda host,port:checks.append((host,port)))
    monkeypatch.setattr(launcher,'ensure_lan_token',lambda *args:'private-lan-token')
    monkeypatch.setattr(launcher,'lan_candidates',lambda:[('en0','192.168.4.21')])
    monkeypatch.setattr(launcher.os,'chdir',lambda path:None)
    monkeypatch.setattr(launcher.os,'execv',lambda executable,args:commands.append(args))
    launcher.launch(tmp_path,True,8001)
    assert checks==[('0.0.0.0',8001)]
    assert commands[0][1:]==['-m','uvicorn','app.main:app','--host','0.0.0.0','--port','8001','--no-access-log']
    connection=(tmp_path/'.runtime/phone-connection.txt').read_text(encoding='utf-8')
    assert 'http://192.168.4.21:8001' in connection and 'private-lan-token' in connection
    output=capsys.readouterr().out
    assert 'http://192.168.4.21:8001' in output and 'API port 8001' in output
    assert ':8000' not in output and ':8000' not in connection and 'private-lan-token' not in output


@pytest.mark.parametrize('options,port',[([],8000),(['--port','8001'],8001)])
def test_main_preserves_port_when_reexecuting_in_venv(tmp_path,monkeypatch,options,port):
    script=tmp_path/'scripts/start_backend.py'
    python=tmp_path/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.touch()
    checks=[]
    commands=[]
    monkeypatch.setattr(launcher,'__file__',str(script))
    monkeypatch.setattr(launcher,'ensure_port_available',lambda host,port:checks.append((host,port)))
    monkeypatch.setattr(launcher.os,'execv',lambda executable,args:commands.append((executable,args)))
    assert launcher.main(['--lan',*options])==0
    assert checks==[('0.0.0.0',port)]
    assert commands==[(str(python),[str(python),str(script),'--_venv','--port',str(port),'--lan'])]


def test_venv_entry_forwards_custom_port_to_launch(monkeypatch):
    calls=[]
    monkeypatch.setattr(launcher,'launch',lambda root,lan,port:calls.append((root,lan,port)) or 0)
    assert launcher.main(['--_venv','--lan','--port','8001'])==0
    assert calls==[(path.parents[1],True,8001)]


@pytest.mark.parametrize('value',['0','65536','-1','invalid'])
def test_invalid_cli_port_rejected_before_setup(monkeypatch,value):
    monkeypatch.setattr(launcher,'ensure_port_available',lambda *args:pytest.fail('Invalid port must not reach setup'))
    with pytest.raises(SystemExit) as error:
        launcher.main(['--port',value])
    assert error.value.code==2
