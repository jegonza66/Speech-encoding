from typing import Optional, Union
from datetime import datetime
from typing import Dict
import requests
import config
import os

def tel_message(
    api_token: str, 
    chat_id: str, 
    message: str = 'Script finished', 
    image: Optional[Union[str, bytes]] = None, 
    caption: Optional[str] = None,
    verbose: bool = True,
    logger:any = None
    ) -> bool:
    '''
    Sends Telegram message with image support.
    
    Args:
        api_token: Bot token (BotFather)
        chat_id: Target chat ID
        message: Text to send
        image: File path or image bytes
        caption: Text that accompanies the image
        verbose: Show status messages
    
    Returns:
        bool: True if sent correctly
    '''
    printfunc = logger.info if logger is not None else print
    try:
        if image is not None:
            return _send_image(api_token, chat_id, image, caption, verbose, printfunc)
        else:
            return _send_text(api_token, chat_id, message, verbose, printfunc)
    except Exception as e:
        if verbose:
            printfunc(f"Error: {e}")
        return False

def _send_text(api_token: str, chat_id: str, message: str, verbose: bool, printfunc):
    """Sends text message"""
    url = f'https://api.telegram.org/bot{api_token}/sendMessage'
    response = requests.post(url, json={'chat_id': chat_id, 'text': message})
    
    success = response.status_code == 200
    if verbose:
        printfunc("✓ Message sent" if success else f"✗ Error: {response.status_code}")
    return success

def _send_image(api_token: str, chat_id: str, image: Union[str, bytes], 
                   caption: Optional[str], verbose: bool, printfunc):
    """Sends image with optional caption"""
    url = f'https://api.telegram.org/bot{api_token}/sendPhoto'
    data = {'chat_id': chat_id}
    if caption:
        data['caption'] = caption
    
    # Prepare file
    if isinstance(image, str):
        if not os.path.exists(image):
            if verbose:
                printfunc(f"✗ File not found: {image}")
            return False
        with open(image, 'rb') as f:
            files = {'photo': f}
            response = requests.post(url, data=data, files=files)
    
    elif isinstance(image, bytes):
        files = {'photo': ('image.png', image, 'image/png')}
        response = requests.post(url, data=data, files=files)
    
    else:
        if verbose:
            printfunc("✗ Invalid image format")
        return False
    
    success = response.status_code == 200
    if verbose:
        printfunc("✓ Image sent" if success else f"✗ Error: {response.status_code}")
    return success


def generate_completion_message(
    situation: str,
    total_number_of_subjects: int,
    stimulus_runtimes: Dict[str, str],
    total_runtime: str,
    save_path: Optional[str] = None,
    fig_path: Optional[str] = None,
) -> str:
    """
    Generate a formatted completion message for the analysis.
    
    Args:
        situation: The current situation being analyzed
        total_number_of_subjects: Total number of subjects processed
        stimulus_runtimes: Dictionary with stimulus runtimes
        total_runtime: Total runtime for the entire analysis
    
    Returns:
        Formatted completion message string
    """
    text = f"""\n
✅ ANÁLISIS COMPLETADO\n

📋 PARÁMETROS:
• Modelo: {config.model}
• Bandas: {config.bands}
• Estímulos: {config.stimuli}
• Condición: {situation}
• Misma regularización entre sujetos: {config.same_validation_subjects}
• Tiempo: ({config.tmin}, {config.tmax})s
• Sujetos: {total_number_of_subjects}/18
• Sesiones: {config.sessions}

\n⚙️ \t CONFIGURACIÓN:
• Folds: {config.n_folds} | Delays: {config.delays[0]}, ...,{config.delays[-1]} | SR: {config.sr} Hz"""

    if config.statistical_test:
        text += f"\n• Test estadístico: ✓ (p < {config.significance})"
    else:
        text += f"\n• Test estadístico: ✗"

    if config.perform_tfce:
        text += f"\n• TFCE: ✓ ({config.n_permutations} perm.)"
    else:
        text += f"\n• TFCE: ✗"

    if config.just_load_data:
        text += f"\n• Modo: 📁 Solo carga"
    else:
        text += f"\n• Modo: 🔬 Completo"

    text += f"""

\n⏱️ \t TIEMPOS DE EJECUCIÓN:"""
    for stim_key, runtime in stimulus_runtimes.items():
        text += f"\n• {stim_key}: {runtime}"
    
    text += f"""
• Total: {total_runtime}

\n📂 Resultados: {save_path}
🎨 Figuras: {fig_path}
\n📅 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    return text


def generate_permutation_completion_message(situation, total_permutations_run, stimulus_runtimes, total_runtime):
    """Generate completion message for random permutations analysis"""
    message = f"""
🎲\t RANDOM PERMUTATIONS ANALYSIS COMPLETED

📊\t PARAMETERS:
• Model: {config.model}
• Bands: {config.bands}
• Stimuli: {config.stimuli}
• Condition: {situation}
• Time interval: ({config.tmin},{config.tmax})s
• Sessions: {config.sessions}
• Permutations: {config.random_permutations}

📈\t RESULTS:
• Total permutations run: {total_permutations_run}
• Subjects processed: {len(config.sessions) * 2}

⏱️\t RUNTIME BREAKDOWN:"""
    
    for stim_band, runtime in stimulus_runtimes.items():
        message += f"\n• {stim_band}: {runtime}"
    
    message += f"""

🏁 TOTAL RUNTIME: {total_runtime}

📁 Script: random_permutations.py
📅 Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    return message
