from marqual_st.cli import main


def test_init_and_validate(tmp_path, capsys):
    p, m = str(tmp_path / "params.csv"), str(tmp_path / "marker_sets.csv")
    assert main(["init-config", "-o", str(tmp_path)]) == 0
    assert main(["init-config", "-o", str(tmp_path)]) == 1     # no overwrite without --force
    assert main(["validate", "-p", p, "-m", m]) == 2           # template paths (/path/to/...) not edited
    assert "input not found" in capsys.readouterr().err
    (tmp_path / "in.h5ad").write_text("x")
    text = (tmp_path / "params.csv").read_text().replace("/path/to/SAMPLE01.tissue.bin50.h5ad", "in.h5ad")
    (tmp_path / "params.csv").write_text(text.replace("/path/to/results/SAMPLE01", "run"))
    assert main(["validate", "-p", p, "-m", m]) == 0
    out = capsys.readouterr().out
    assert "OK: sample" in out and str(tmp_path / "run" / "figures") in out


def test_bad_config(tmp_path, capsys):
    (tmp_path / "p.csv").write_text("section,parameter,value\nrun,sample,x\nrun,input,y\nrun,outdir,z\n"
                                    "run,platform,xenium\n")
    (tmp_path / "m.csv").write_text("a,G1\n")
    assert main(["validate", "-p", str(tmp_path / "p.csv"), "-m", str(tmp_path / "m.csv")]) == 2
    assert "platform" in capsys.readouterr().err
    assert main(["validate", "-p", str(tmp_path / "nope.csv"), "-m", str(tmp_path / "m.csv")]) == 2
    assert "params file not found" in capsys.readouterr().err
