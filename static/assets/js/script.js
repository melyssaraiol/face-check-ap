function iniciarReconhecimentoFacial(modulo) {
        const cameraId = cameras[modulo];  // Obtém o número da câmera com base no módulo

        // Redireciona para a página de reconhecimento com o ID da câmera como parâmetro
        window.location.href = `/reconhecimento?camera=${modulo}`;
}