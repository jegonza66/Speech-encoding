from typing import Optional, Union
import requests
import os

def tel_message(
    api_token: str, 
    chat_id: str, 
    message: str = 'Script finished', 
    image: Optional[Union[str, bytes]] = None, 
    caption: Optional[str] = None,
    verbose: bool = True
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
    
    try:
        if image is not None:
            return _send_image(api_token, chat_id, image, caption, verbose)
        else:
            return _send_text(api_token, chat_id, message, verbose)
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return False

def _send_text(api_token: str, chat_id: str, message: str, verbose: bool):
    """Sends text message"""
    url = f'https://api.telegram.org/bot{api_token}/sendMessage'
    response = requests.post(url, json={'chat_id': chat_id, 'text': message})
    
    success = response.status_code == 200
    if verbose:
        print("✅ Message sent" if success else f"❌ Error: {response.status_code}")
    return success

def _send_image(api_token: str, chat_id: str, image: Union[str, bytes], 
                   caption: Optional[str], verbose: bool):
    """Sends image with optional caption"""
    url = f'https://api.telegram.org/bot{api_token}/sendPhoto'
    data = {'chat_id': chat_id}
    if caption:
        data['caption'] = caption
    
    # Prepare file
    if isinstance(image, str):
        if not os.path.exists(image):
            if verbose:
                print(f"❌ File not found: {image}")
            return False
        with open(image, 'rb') as f:
            files = {'photo': f}
            response = requests.post(url, data=data, files=files)
    
    elif isinstance(image, bytes):
        files = {'photo': ('image.png', image, 'image/png')}
        response = requests.post(url, data=data, files=files)
    
    else:
        if verbose:
            print("❌ Invalid image format")
        return False
    
    success = response.status_code == 200
    if verbose:
        print("✅ Image sent" if success else f"❌ Error: {response.status_code}")
    return success