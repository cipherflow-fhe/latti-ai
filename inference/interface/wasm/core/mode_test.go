package core

import "testing"

func TestDeploymentModeBinding(t *testing.T) {
	valid := []Task{
		{},
		{DeploymentMode: "client_encrypted_input", InputMode: "client_ciphertext", ParameterMode: "plaintext_lazy", Decryptor: "client"},
		{DeploymentMode: "server_provisioned_runner", InputMode: "plaintext", ParameterMode: "encrypted_offline", Decryptor: "provisioner"},
	}
	for _, task := range valid {
		if err := task.validateMode(); err != nil {
			t.Fatal(err)
		}
	}
	for _, task := range []Task{
		{DeploymentMode: "unknown"},
		{InputMode: "plaintext"},
		{DeploymentMode: "client_encrypted_input", InputMode: "plaintext", ParameterMode: "encrypted_offline", Decryptor: "provisioner"},
		{DeploymentMode: "server_provisioned_runner", InputMode: "client_ciphertext", ParameterMode: "plaintext_lazy", Decryptor: "client"},
	} {
		if task.validateMode() == nil {
			t.Fatal("accepted inconsistent deployment mode", task)
		}
	}
	client := &Client{Task: valid[2]}
	if _, err := client.StartEncrypt("crossing", 8); err == nil {
		t.Fatal("parameter mode used Feature encryption")
	}
}
