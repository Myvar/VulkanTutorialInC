#include <assimp/mesh.h>
#include <cglm/affine.h>
#include <cglm/mat3.h>
#include <cglm/mat4.h>
#include <cglm/types.h>
#include <cglm/vec2.h>
#include <stddef.h>

#include <assimp/cimport.h>     // Plain-C interface
#include <assimp/postprocess.h> // Post processing flag
#include <assimp/scene.h>       // Output data structure
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define CGLM_FORCE_DEPTH_ZERO_TO_ONE
#include <cglm/cglm.h>
#include <main.h>

#define max(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

// would be nice to define this per shader or something or modle idk
typedef struct Vertex {
  vec3 pos;
  vec3 color;
  vec2 texCoord;
} Vertex;

typedef struct UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;

} UniformBufferObject;

Vertex *vertices = NULL;
uint32_t *indices = NULL;

static VkVertexInputBindingDescription get_binding_description() {
  VkVertexInputBindingDescription bindingDescription = {0};
  bindingDescription.binding = 0;
  bindingDescription.stride = sizeof(Vertex);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  return bindingDescription;
}

static VkVertexInputAttributeDescription *get_attribute_descriptions() {
  VkVertexInputAttributeDescription *attributeDescriptions = NULL;
  arrsetlen(attributeDescriptions, 3);

  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, pos);

  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, color);

  attributeDescriptions[2].binding = 0;
  attributeDescriptions[2].location = 2;
  attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

  return attributeDescriptions;
}

int screenWidth = 800;
int screenHeight = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;
uint32_t currentFrame = 0;

typedef struct QueueFamilyIndices_t {
  uint32_t graphics_family;
  bool graphics_family_present;

  uint32_t present_family;
  bool present_family_present;
} QueueFamilyIndices;

typedef struct swap_chain_support_details_t {
  VkSurfaceCapabilitiesKHR capabilities;
  VkSurfaceFormatKHR *formats;
  VkPresentModeKHR *presentModes;
} swap_chain_support_details;

GLFWwindow *window = NULL;
VkInstance instance = {0};
char **layers = NULL;
VkDebugUtilsMessengerEXT debugMessenger = {0};
VkPhysicalDevice physicalDevice = NULL;

QueueFamilyIndices qfi = {0};
VkDevice device;
VkQueue graphicsQueue;
VkSurfaceKHR surface;
VkQueue presentQueue;
VkSwapchainKHR swapChain;
VkImage *swapChainImages = NULL;
VkFormat swapChainImageFormat;
VkExtent2D swapChainExtent;
VkImageView *swapChainImageViews = NULL;
VkPipelineLayout pipelineLayout;
VkRenderPass renderPass;
VkPipeline graphicsPipeline;
VkFramebuffer *swapChainFramebuffers = NULL;
VkCommandPool commandPool;
VkCommandBuffer *commandBuffers = NULL;
VkSemaphore *imageAvailableSemaphores = NULL;
VkSemaphore *renderFinishedSemaphores = NULL;
VkFence *inFlightFences = NULL;
bool framebufferResized = false;
VkBuffer vertexBuffer;
VkDeviceMemory vertexBufferMemory;
VkBuffer indexBuffer;
VkDeviceMemory indexBufferMemory;
VkDescriptorSetLayout descriptorSetLayout;
VkBuffer *uniformBuffers = NULL;
VkDeviceMemory *uniformBuffersMemory = NULL;
void **uniformBuffersMapped = NULL;
VkDescriptorPool descriptorPool;
VkDescriptorSet *descriptorSets = NULL;
uint32_t mipLevels;
VkImage textureImage;
VkDeviceMemory textureImageMemory;
VkImageView textureImageView;
VkSampler textureSampler;
VkImage depthImage;
VkDeviceMemory depthImageMemory;
VkImageView depthImageView;
VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

VkImage colorImage;
VkDeviceMemory colorImageMemory;
VkImageView colorImageView;

VkSampleCountFlagBits get_max_usable_sample_count() {
  VkPhysicalDeviceProperties physicalDeviceProperties;
  vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

  VkSampleCountFlags counts =
      physicalDeviceProperties.limits.framebufferColorSampleCounts &
      physicalDeviceProperties.limits.framebufferDepthSampleCounts;
  if (counts & VK_SAMPLE_COUNT_64_BIT) {
    return VK_SAMPLE_COUNT_64_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_32_BIT) {
    return VK_SAMPLE_COUNT_32_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_16_BIT) {
    return VK_SAMPLE_COUNT_16_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_8_BIT) {
    return VK_SAMPLE_COUNT_8_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_4_BIT) {
    return VK_SAMPLE_COUNT_4_BIT;
  }
  if (counts & VK_SAMPLE_COUNT_2_BIT) {
    return VK_SAMPLE_COUNT_2_BIT;
  }

  return VK_SAMPLE_COUNT_1_BIT;
}

bool check_extension_support() {
  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
  VkExtensionProperties *extensions = NULL;
  arrsetlen(extensions, extensionCount);
  vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensions);
  for (int i = 0; i < arrlen(extensions); i++)
    printf("Found Ext: '%s' v%d\n", extensions[i].extensionName,
           extensions[i].specVersion);
  arrfree(extensions);

  return false;
}

bool check_validation_layer_support() {
  bool flag = false;

  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, NULL);
  VkLayerProperties *availableLayers = NULL;
  arrsetlen(availableLayers, layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);

  for (int i = 0; i < arrlen(availableLayers); i++) {
    printf("Found Layer: '%s - %s'\n", availableLayers[i].layerName,
           availableLayers[i].description);
    if (strcmp(availableLayers[i].layerName, "VK_LAYER_KHRONOS_validation") ==
        0)
      flag = true;
  }

  arrfree(availableLayers);
  return flag;
}

char **get_required_extensions() {
  uint32_t glfwExtensionCount = 0;
  char **glfwExtensions =
      (char **)glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  char **extensions = NULL;
  for (int i = 0; i < glfwExtensionCount; i++) {
    printf("Enableing GLFW EXT: %s\n", glfwExtensions[i]);
    arrput(extensions, glfwExtensions[i]);
  }

  arrput(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  return extensions;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
               VkDebugUtilsMessageTypeFlagsEXT message_type,
               const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
               void *user_data) {

  if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    // Message is important enough to show
    printf("validation layer: %s\n", callback_data->pMessage);
  }

  return VK_FALSE;
}

VkResult create_debug_utils_messenger_ext(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *create_info,
    const VkAllocationCallbacks *allocator,
    VkDebugUtilsMessengerEXT *debug_messenger) {
  PFN_vkCreateDebugUtilsMessengerEXT func =
      (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != NULL) {
    return func(instance, create_info, allocator, debug_messenger);

  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void setup_debug_messager() {
  VkDebugUtilsMessengerCreateInfoEXT createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debug_callback;
  createInfo.pUserData = NULL;

  if (create_debug_utils_messenger_ext(instance, &createInfo, NULL,
                                       &debugMessenger) != VK_SUCCESS) {
    perror("failed to set up debug messenger!");
  }
}

void destroy_debug_utils_messenger_ext(VkInstance instance,
                                       VkDebugUtilsMessengerEXT debug_messenger,
                                       const VkAllocationCallbacks *allocator) {
  PFN_vkDestroyDebugUtilsMessengerEXT func =
      (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != NULL) {
    func(instance, debug_messenger, allocator);
  }
}

bool is_queue_family_indices_complete(QueueFamilyIndices q) {
  return q.graphics_family_present && q.present_family_present;
}

QueueFamilyIndices find_queue_families(VkPhysicalDevice device) {
  QueueFamilyIndices indices = {0};

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, NULL);
  VkQueueFamilyProperties *queueFamilies = NULL;
  arrsetlen(queueFamilies, queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies);

  for (int i = 0; i < queueFamilyCount; i++) {
    VkQueueFamilyProperties qfp = queueFamilies[i];
    if (qfp.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphics_family = i;
      indices.graphics_family_present = true;
    }

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
    if (presentSupport) {
      indices.present_family = i;
      indices.present_family_present = true;
      printf("Found present layer: %d\n", i);
    }

    if (is_queue_family_indices_complete(indices))
      break;
  }

  return indices;
}

swap_chain_support_details query_swap_chain_support(VkPhysicalDevice device) {
  swap_chain_support_details details = {0};

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                            &details.capabilities);

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, NULL);

  if (formatCount != 0) {
    arrsetlen(details.formats, formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         details.formats);
  }

  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                            NULL);
  if (presentModeCount != 0) {
    arrsetlen(details.presentModes, presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &presentModeCount, details.presentModes);
  }

  return details;
}

bool is_device_suitable(VkPhysicalDevice device) {
  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
  bool queueFamilyComplete = is_queue_family_indices_complete(qfi);

  printf("Found physical device: %s - is descrete %s - geom support %s - "
         "queue family %s\n",
         deviceProperties.deviceName,
         deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
             ? "true"
             : "false",
         deviceFeatures.geometryShader ? "true" : "false",
         queueFamilyComplete ? "true" : "false");

  bool extFlag = false;
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, NULL);
  VkExtensionProperties *availableExtensions = NULL;
  arrsetlen(availableExtensions, extensionCount);
  vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount,
                                       availableExtensions);

  for (int i = 0; i < extensionCount; i++) {
    VkExtensionProperties ext = availableExtensions[i];
    if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
      extFlag = true;
    }
  }

  bool swapchainFlag = false;
  if (extFlag) {
    swap_chain_support_details swapChainSupport =
        query_swap_chain_support(device);
    swapchainFlag = arrlen(swapChainSupport.formats) > 0 &&
                    arrlen(swapChainSupport.presentModes) > 0;
  }
  VkPhysicalDeviceFeatures supportedFeatures;
  vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

  return deviceFeatures.geometryShader && queueFamilyComplete && extFlag &&
         swapchainFlag && supportedFeatures.samplerAnisotropy;
  // return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
  // && deviceFeatures.geometryShader;
}

VkSurfaceFormatKHR
choose_swap_surface_format(VkSurfaceFormatKHR *available_formats) {
  for (int i = 0; i < arrlen(available_formats); i++) {
    VkSurfaceFormatKHR availableFormat = available_formats[i];
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
        availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }
  return available_formats[0];
}

VkPresentModeKHR
choose_swap_present_mode(VkPresentModeKHR *available_present_modes) {
  for (int i = 0; i < arrlen(available_present_modes); i++) {
    VkPresentModeKHR availablePresentMode = available_present_modes[i];
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    }
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}

int clamp(int value, int min, int max) {
  if (value < min) {
    return min;
  } else if (value > max) {
    return max;
  }
  return value;
}

VkExtent2D choose_swap_extent(VkSurfaceCapabilitiesKHR capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;

  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D actualExtent = {(uint32_t)(width), (uint32_t)(height)};
    actualExtent.width =
        clamp(actualExtent.width, capabilities.minImageExtent.width,
              capabilities.maxImageExtent.width);
    actualExtent.height =
        clamp(actualExtent.height, capabilities.minImageExtent.height,
              capabilities.maxImageExtent.height);
    return actualExtent;
  }
}

void pick_physical_device() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
  if (deviceCount == 0) {
    perror("failed to find GPUs with Vulkan support");
  }

  VkPhysicalDevice *devices = NULL;
  arrsetlen(devices, deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

  bool flag = false;
  for (int i = 0; i < arrlen(devices); i++) {
    VkPhysicalDevice pd = devices[i];
    qfi = find_queue_families(pd);
    if (is_device_suitable(pd)) {

      physicalDevice = pd;
      msaaSamples = get_max_usable_sample_count();
      // BE GONE JAGGIES!!!!
      // msaaSamples = VK_SAMPLE_COUNT_1_BIT;
      printf("MSAA: %d\n", msaaSamples);
      flag = true;
      break;
    }
    qfi = (QueueFamilyIndices){0};
  }

  if (!flag) {
    printf("Failed to find usable vulkan gpu\n");
    exit(EXIT_FAILURE);
  }
}

void create_logical_device() {

  VkDeviceQueueCreateInfo queueInfos[2] = {0};
  float queuePriority = 1.0f;

  queueInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfos[0].queueFamilyIndex = qfi.graphics_family;
  queueInfos[0].queueCount = 1;
  queueInfos[0].pQueuePriorities = &queuePriority;

  queueInfos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfos[1].queueFamilyIndex = qfi.present_family;
  queueInfos[1].queueCount = 1;
  queueInfos[1].pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures = {0};
  deviceFeatures.samplerAnisotropy = VK_TRUE;
  deviceFeatures.sampleRateShading = VK_TRUE; // enable sample

  VkDeviceCreateInfo createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pQueueCreateInfos = queueInfos;
  //  If the queue families are the same, then we only need to pass its index
  //  once.
  createInfo.queueCreateInfoCount =
      (qfi.graphics_family == qfi.present_family ? 1 : 2);
  createInfo.pEnabledFeatures = &deviceFeatures;

  const char *deviceExtensions[] = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      // Add other device extensions if needed
  };

  createInfo.enabledExtensionCount =
      sizeof(deviceExtensions) / sizeof(deviceExtensions[0]);
  createInfo.ppEnabledExtensionNames = deviceExtensions;

  /*
   Previous implementations of Vulkan made a distinction between instance and
   device specific validation layers, but this is no longer the case. That
   means that the enabledLayerCount and ppEnabledLayerNames fields of
   VkDeviceCreateInfo are ignored by up-to-date implementations. However, it
   is still a good idea to set them anyway to be compatible with older
   implementations:
   */
  /*if (!check_validation_layer_support()) {
    printf("validation support not avilible\n");
    create_info.enabledLayerCount = 0;
  } else {
    char *layer = "VK_LAYER_KHRONOS_validation";
    arrput(layers, layer);
    create_info.ppEnabledLayerNames = (const char *const *)layers;
    create_info.enabledLayerCount = arrlen(layers);
    printf("enabled validation layer: [%d]%s\n", create_info.enabledLayerCount,
           create_info.ppEnabledLayerNames[0]);
  }*/

  if (vkCreateDevice(physicalDevice, &createInfo, NULL, &device) !=
      VK_SUCCESS) {
    perror("failed to create logical device!");
  }
}

void create_queues_for_device(VkDevice device) {
  vkGetDeviceQueue(device, qfi.graphics_family, 0, &graphicsQueue);
  vkGetDeviceQueue(device, qfi.present_family, 0, &presentQueue);
}

void create_surface() {
  if (glfwCreateWindowSurface(instance, window, NULL, &surface) != VK_SUCCESS) {
    perror("failed to create window surface!");
    exit(EXIT_FAILURE);
  }
}

void create_swap_chain() {
  swap_chain_support_details swapChainSupport =
      query_swap_chain_support(physicalDevice);
  VkSurfaceFormatKHR surfaceFormat =
      choose_swap_surface_format(swapChainSupport.formats);
  VkPresentModeKHR presentMode =
      choose_swap_present_mode(swapChainSupport.presentModes);
  VkExtent2D extent = choose_swap_extent(swapChainSupport.capabilities);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface;

  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queueFamilyIndices[] = {qfi.graphics_family, qfi.present_family};

  if (qfi.graphics_family != qfi.present_family) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;  // Optional
    createInfo.pQueueFamilyIndices = NULL; // Optional
  }

  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;

  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(device, &createInfo, NULL, &swapChain) !=
      VK_SUCCESS) {
    perror("failed to create swap chain!");
    exit(EXIT_FAILURE);
  }

  vkGetSwapchainImagesKHR(device, swapChain, &imageCount, NULL);
  arrsetlen(swapChainImages, imageCount);
  vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages);

  swapChainImageFormat = surfaceFormat.format;
  swapChainExtent = extent;
}

VkImageView create_image_view(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags,
                              uint32_t mipLevels) {
  VkImageViewCreateInfo viewInfo = {0};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;
  viewInfo.subresourceRange.levelCount = mipLevels;

  VkImageView imageView;
  if (vkCreateImageView(device, &viewInfo, NULL, &imageView) != VK_SUCCESS) {
    perror("failed to create texture image view!");
    exit(EXIT_FAILURE);
  }

  return imageView;
}

void create_image_views() {
  arrsetlen(swapChainImageViews, arrlen(swapChainImages));

  for (size_t i = 0; i < arrlen(swapChainImages); i++) {
    swapChainImageViews[i] = create_image_view(
        swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
  }
}

char *read_file(const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("failed to open file");
    return NULL;
  }

  fseek(file, 0, SEEK_END);
  long fileSize = ftell(file);
  fseek(file, 0, SEEK_SET);

  char *buffer = NULL;
  arrsetlen(buffer, fileSize);
  if (!buffer) {
    fclose(file);
    perror("failed to allocate memory");
    return NULL;
  }

  fread(buffer, 1, fileSize, file);
  fclose(file);

  return buffer;
}

VkShaderModule create_shader_module(char *code) {
  VkShaderModuleCreateInfo createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = arrlen(code);
  createInfo.pCode = (uint32_t *)(code);

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, NULL, &shaderModule) !=
      VK_SUCCESS) {
    perror("failed to create shader module!");
    exit(0);
  }
  return shaderModule;
}

void create_graphics_pipeline() {
  char *vertShaderCode = read_file("shaders/min.vert.spv");
  char *fragShaderCode = read_file("shaders/min.frag.spv");

  VkShaderModule vertShaderModule = create_shader_module(vertShaderCode);
  VkShaderModule fragShaderModule = create_shader_module(fragShaderCode);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo = {0};
  vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo = {0};
  fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                    fragShaderStageInfo};

  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {0};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkVertexInputBindingDescription bindingDescription =
      get_binding_description();
  VkVertexInputAttributeDescription *attributeDescriptions =
      get_attribute_descriptions();

  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;

  vertexInputInfo.vertexAttributeDescriptionCount =
      arrlen(attributeDescriptions);
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {0};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport = {0};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)swapChainExtent.width;
  viewport.height = (float)swapChainExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor = {0};
  scissor.offset = (VkOffset2D){0, 0};
  scissor.extent = swapChainExtent;

  VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                    VK_DYNAMIC_STATE_SCISSOR};

  VkPipelineDynamicStateCreateInfo dynamicState = {0};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount =
      (uint32_t)((sizeof dynamicStates / sizeof dynamicStates[0]));
  dynamicState.pDynamicStates = dynamicStates;

  VkPipelineViewportStateCreateInfo viewportState = {0};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer = {0};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  // rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling = {0};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = msaaSamples;
  multisampling.sampleShadingEnable = VK_TRUE; // enable sample
  multisampling.minSampleShading =
      .2f; // min fraction for sample shading; closer to one is smoother

  VkPipelineColorBlendAttachmentState colorBlendAttachment = {0};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  colorBlendAttachment.blendEnable = VK_FALSE;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
  colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;             // Optional
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;             // Optional

  colorBlendAttachment.blendEnable = VK_TRUE;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorBlendAttachment.dstColorBlendFactor =
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo colorBlending = {0};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f; // Optional
  colorBlending.blendConstants[1] = 0.0f; // Optional
  colorBlending.blendConstants[2] = 0.0f; // Optional
  colorBlending.blendConstants[3] = 0.0f; // Optional

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
  pipelineLayoutInfo.pPushConstantRanges = NULL; // Optional
  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL,
                             &pipelineLayout) != VK_SUCCESS) {
    perror("failed to create pipeline layout!");
  }

  VkPipelineDepthStencilStateCreateInfo depthStencil = {0};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.minDepthBounds = 0.0f; // Optional
  depthStencil.maxDepthBounds = 1.0f; // Optional
  depthStencil.stencilTestEnable = VK_FALSE;

  VkGraphicsPipelineCreateInfo pipelineInfo = {0};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;

  pipelineInfo.pDepthStencilState = &depthStencil;

  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = renderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
  pipelineInfo.basePipelineIndex = -1;              // Optional

  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL,
                                &graphicsPipeline) != VK_SUCCESS) {
    perror("failed to create graphics pipeline!");
  }

  vkDestroyShaderModule(device, fragShaderModule, NULL);
  vkDestroyShaderModule(device, vertShaderModule, NULL);
  arrfree(attributeDescriptions);
}

VkCommandBuffer begin_single_time_commands() {
  VkCommandBufferAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {0};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

void end_single_time_commands(VkCommandBuffer commandBuffer) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo = {0};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);
  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
  VkCommandBuffer commandBuffer = begin_single_time_commands();

  VkBufferCopy copyRegion = {0};
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

  end_single_time_commands(commandBuffer);
}

bool has_stencil_component(VkFormat format) {
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
         format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void transition_image_layout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             uint32_t mipLevels) {
  VkCommandBuffer commandBuffer = begin_single_time_commands();

  VkImageMemoryBarrier barrier = {0};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;

  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  barrier.image = image;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.subresourceRange.levelCount = mipLevels;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (has_stencil_component(format)) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  } else {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  }

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else {
    perror("unsupported layout transition!");
    exit(EXIT_FAILURE);
  }

  vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, NULL,
                       0, NULL, 1, &barrier);

  end_single_time_commands(commandBuffer);
}

VkFormat find_supported_format(VkFormat *candidates, int can_len,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features) {
  for (int i = 0; i < can_len; i++) {
    VkFormat format = candidates[i];
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;

    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  perror("Failed to find a supported format");
  exit(EXIT_FAILURE);
}

VkFormat find_depth_format() {
  VkFormat canidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                          VK_FORMAT_D24_UNORM_S8_UINT};
  return find_supported_format(
      canidates, (sizeof canidates / sizeof canidates[0]),
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

uint32_t find_memory_type(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  perror("failed to find suitable memory type!");
  exit(EXIT_FAILURE);
}

void create_image(uint32_t width, uint32_t height, uint32_t mipLevels,
                  VkSampleCountFlagBits numSamples, VkFormat format,
                  VkImageTiling tiling, VkImageUsageFlags usage,
                  VkMemoryPropertyFlags properties, VkImage *image,
                  VkDeviceMemory *imageMemory) {
  VkImageCreateInfo imageInfo = {0};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.mipLevels = mipLevels;
  imageInfo.samples = numSamples;

  if (vkCreateImage(device, &imageInfo, NULL, image) != VK_SUCCESS) {
    perror("failed to create image!");
    exit(EXIT_FAILURE);
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, *image, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, NULL, imageMemory) != VK_SUCCESS) {
    perror("failed to allocate image memory!");
    exit(EXIT_FAILURE);
  }

  vkBindImageMemory(device, *image, *imageMemory, 0);
}

void create_depth_resources() {
  VkFormat depthFormat = find_depth_format();
  create_image(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
               depthFormat, VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &depthImage,
               &depthImageMemory);
  depthImageView =
      create_image_view(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
  transition_image_layout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
}

void create_render_pass() {
  VkAttachmentDescription colorAttachment = {0};
  colorAttachment.format = swapChainImageFormat;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.samples = msaaSamples;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentDescription colorAttachmentResolve = {0};
  colorAttachmentResolve.format = swapChainImageFormat;
  colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef = {0};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference colorAttachmentResolveRef = {0};
  colorAttachmentResolveRef.attachment = 2;
  colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentDescription depthAttachment = {0};
  depthAttachment.format = find_depth_format();
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAttachment.samples = msaaSamples;

  VkAttachmentReference depthAttachmentRef = {0};
  depthAttachmentRef.attachment = 1;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {0};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;
  subpass.pResolveAttachments = &colorAttachmentResolveRef;

  VkSubpassDependency dependency = {0};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  VkAttachmentDescription attachments[] = {colorAttachment, depthAttachment,
                                           colorAttachmentResolve};
  VkRenderPassCreateInfo renderPassInfo = {0};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = (sizeof attachments / sizeof attachments[0]);
  renderPassInfo.pAttachments = attachments;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  if (vkCreateRenderPass(device, &renderPassInfo, NULL, &renderPass) !=
      VK_SUCCESS) {
    perror("failed to create render pass!");
    exit(EXIT_FAILURE);
  }
}

void create_framebuffers() {
  arrsetlen(swapChainFramebuffers, arrlen(swapChainImageViews));
  for (size_t i = 0; i < arrlen(swapChainImageViews); i++) {
    VkImageView attachments[] = {colorImageView, depthImageView,
                                 swapChainImageViews[i]};

    VkFramebufferCreateInfo framebufferInfo = {0};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount =
        (sizeof attachments / sizeof attachments[0]);
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = swapChainExtent.width;
    framebufferInfo.height = swapChainExtent.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, NULL,
                            &swapChainFramebuffers[i]) != VK_SUCCESS) {
      perror("failed to create framebuffer!");
      exit(EXIT_FAILURE);
    }
  }
}

void create_command_pool() {

  /*
   There are two possible flags for command pools:
    • VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers
      are rerecorded with new commands very often (may change memory allocation
   behavior)

   • VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: Allow command
   buffers to be rerecorded individually, without this flag they all have to be
   reset together
  */

  VkCommandPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = qfi.graphics_family;

  if (vkCreateCommandPool(device, &poolInfo, NULL, &commandPool) !=
      VK_SUCCESS) {
    perror("failed to create command pool!");
    exit(EXIT_FAILURE);
  }
}

void create_command_buffers() {

  /*
    The level parameter specifies if the allocated command buffers are primary
    or secondary command buffers.

    • VK_COMMAND_BUFFER_LEVEL_PRIMARY: Can be
    submitted to a queue for execution, but cannot be called from other command
    buffers.

    • VK_COMMAND_BUFFER_LEVEL_SECONDARY: Cannot be submitted directly,
    but can be called from primary command buffers.


    We won’t make use of the secondary command buffer functionality here, but
    you can imagine that it’s helpful to reuse common operations from primary
    command buffers.
  */

  arrsetlen(commandBuffers, MAX_FRAMES_IN_FLIGHT);

  VkCommandBufferAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

  if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers) !=
      VK_SUCCESS) {
    perror("failed to allocate command buffers!");
    exit(EXIT_FAILURE);
  }
}

void record_command_buffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
  VkCommandBufferBeginInfo beginInfo = {0};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0;               // Optional
  beginInfo.pInheritanceInfo = NULL; // Optional
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    perror("failed to begin recording command buffer!");
    exit(EXIT_FAILURE);
  }

  /*
   The flags parameter specifies how we’re going to use the command buffer. The
  following values are available:

  • VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command
  buffer will be rerecorded right after executing it once.

  • VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: This is a secondary
  command buffer that will be entirely within a single render pass.

  • VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: The command
  buffer can be resubmitted while it is also already pending execution
   */
  VkRenderPassBeginInfo renderPassInfo = {0};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = renderPass;
  renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

  renderPassInfo.renderArea.offset = (VkOffset2D){0, 0};
  renderPassInfo.renderArea.extent = swapChainExtent;

  VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearColor;

  VkClearValue clearValues[2] = {0};
  clearValues[0].color = (VkClearColorValue){{0.0f, 0.0f, 0.0f, 1.0f}};
  clearValues[1].depthStencil = (VkClearDepthStencilValue){1.0f, 0};

  renderPassInfo.clearValueCount = (sizeof clearValues / sizeof clearValues[0]);
  renderPassInfo.pClearValues = clearValues;

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphicsPipeline);

  VkViewport viewport = {0};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)(swapChainExtent.width);
  viewport.height = (float)(swapChainExtent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

  VkRect2D scissor = {0};
  scissor.offset = (VkOffset2D){0, 0};
  scissor.extent = swapChainExtent;
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  /*
    • vertexCount: Even though we don’t have a vertex buffer, we technically
    still have 3 vertices to draw.

    • instanceCount: Used for instanced rendering, use 1 if you’re not doing
    that.

    • firstVertex: Used as an offset into the vertex buffer, defines the lowest
    value of gl_VertexIndex.

    • firstInstance: Used as an offset for instanced rendering, defines the
    lowest value of gl_InstanceIndex.
  */

  //  vkCmdDraw(commandBuffer, 3, 1, 0, 0);

  VkBuffer vertexBuffers[] = {vertexBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0, 1, &descriptorSets[currentFrame],
                          0, NULL);
  vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

  // vkCmdDraw(commandBuffer, (sizeof VERTICES / sizeof VERTICES[0]), 1, 0, 0);
  vkCmdDrawIndexed(commandBuffer, (uint32_t)arrlen(indices), 1, 0, 0, 0);
  vkCmdEndRenderPass(commandBuffer);
  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    perror("failed to record command buffer!");
  }
}

void create_sync_objects() {

  arrsetlen(imageAvailableSemaphores, MAX_FRAMES_IN_FLIGHT);
  arrsetlen(renderFinishedSemaphores, MAX_FRAMES_IN_FLIGHT);
  arrsetlen(inFlightFences, MAX_FRAMES_IN_FLIGHT);
  VkSemaphoreCreateInfo semaphoreInfo = {0};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo = {0};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(device, &semaphoreInfo, NULL,
                          &imageAvailableSemaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(device, &semaphoreInfo, NULL,
                          &renderFinishedSemaphores[i]) != VK_SUCCESS ||
        vkCreateFence(device, &fenceInfo, NULL, &inFlightFences[i]) !=
            VK_SUCCESS) {
      perror("failed to create semaphores!");
      exit(EXIT_FAILURE);
    }
  }
}

void cleanup_swap_chain() {
  vkDestroyImageView(device, depthImageView, NULL);
  vkDestroyImage(device, depthImage, NULL);
  vkFreeMemory(device, depthImageMemory, NULL);

  vkDestroyImageView(device, colorImageView, NULL);
  vkDestroyImage(device, colorImage, NULL);
  vkFreeMemory(device, colorImageMemory, NULL);

  for (size_t i = 0; i < arrlen(swapChainFramebuffers); i++) {
    vkDestroyFramebuffer(device, swapChainFramebuffers[i], NULL);
  }
  for (size_t i = 0; i < arrlen(swapChainImageViews); i++) {
    vkDestroyImageView(device, swapChainImageViews[i], NULL);
  }

  vkDestroySwapchainKHR(device, swapChain, NULL);
}

void cleanup_sync_objects() {
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(device, renderFinishedSemaphores[i], NULL);
    vkDestroySemaphore(device, imageAvailableSemaphores[i], NULL);
    vkDestroyFence(device, inFlightFences[i], NULL);
  }
}

void create_color_resources() {
  VkFormat colorFormat = swapChainImageFormat;

  create_image(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
               colorFormat, VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &colorImage,
               &colorImageMemory);
  colorImageView =
      create_image_view(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
}

void recreate_swap_chain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

  screenWidth = width;
  screenHeight = height;

  vkDeviceWaitIdle(device);

  cleanup_swap_chain();
  cleanup_sync_objects();

  create_swap_chain();
  create_image_views();
  create_color_resources();
  create_depth_resources();
  create_framebuffers();
  create_sync_objects();
}

void update_uniform_buffer(uint32_t currentImage) {
  double time = glfwGetTime();
  UniformBufferObject ubo = {0};

  glm_mat4_identity(ubo.model);
  glm_rotate(ubo.model, glm_rad(180) * time * 0.1f, (vec3){1, 1, 1});

  mat4 buf;
  glm_mat4_identity(buf);
  glm_scale(buf, (vec3){1, 1, 1});
  glm_mat4_mul(ubo.model, buf, ubo.model);

  glm_mat4_identity(ubo.view);
  glm_lookat((vec3){2.0f, 0.0f, 2.0f}, (vec3){0.0f, 0.0f, 0.0f},
             (vec3){0.0f, 0.0f, 1.0f}, ubo.view);
  glm_perspective(glm_rad(45.0f),
                  swapChainExtent.width / (float)swapChainExtent.height, 0.1f,
                  10.0f, ubo.proj);

  ubo.proj[1][1] *= -1;

  memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void draw_frame() {
  vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                  UINT64_MAX);

  uint32_t imageIndex;
  VkResult result = vkAcquireNextImageKHR(
      device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame],
      VK_NULL_HANDLE, &imageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      framebufferResized) {
    framebufferResized = false;
    recreate_swap_chain();
    return;

  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    perror("failed to acquire swap chain image!");
    exit(EXIT_FAILURE);
  }
  vkResetFences(device, 1, &inFlightFences[currentFrame]);

  vkResetCommandBuffer(commandBuffers[currentFrame], 0);
  record_command_buffer(commandBuffers[currentFrame], imageIndex);

  update_uniform_buffer(currentFrame);

  VkSubmitInfo submitInfo = {0};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

  VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                    inFlightFences[currentFrame]) != VK_SUCCESS) {
    perror("failed to submit draw command buffer!");
    exit(EXIT_FAILURE);
  }

  VkPresentInfoKHR presentInfo = {0};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;

  VkSwapchainKHR swapChains[] = {swapChain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;
  result = vkQueuePresentKHR(presentQueue, &presentInfo);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    recreate_swap_chain();
  } else if (result != VK_SUCCESS) {
    perror("failed to present swap chain image!");
    exit(EXIT_FAILURE);
  }

  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags properties, VkBuffer *buffer,
                   VkDeviceMemory *bufferMemory) {
  VkBufferCreateInfo bufferInfo = {0};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, NULL, buffer) != VK_SUCCESS) {
    perror("failed to create buffer!");
    exit(EXIT_FAILURE);
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, *buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, NULL, bufferMemory) != VK_SUCCESS) {
    perror("failed to allocate buffer memory!");
    exit(EXIT_FAILURE);
  }

  vkBindBufferMemory(device, *buffer, *bufferMemory, 0);
}

void create_vertex_buffer() {
  /*
    It should be noted that in a real world application, you’re not supposed
    to actually call vkAllocateMemory for every individual buffer. The
    maximum number of simultaneous memory allocations is limited by the
    maxMemoryAllocationCount physical device limit, which may be as low as
    4096 even on high end hardware like an NVIDIA GTX 1080. The right way to
    allocate memory for a large number of objects at the same time is to
    create a custom allocator that splits up a single allocation among many
    different objects by using the offset parameters that we’ve seen in many
    functions.You can either implement such an allocator yourself, or use the
    VulkanMemoryAllocator library provided by the GPUOpen initiative
   */
  VkDeviceSize bufferSize = sizeof(vertices[0]) * arrlen(vertices);
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &stagingBuffer, &stagingBufferMemory);

  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, vertices, (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);
  create_buffer(
      bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertexBuffer, &vertexBufferMemory);
  copy_buffer(stagingBuffer, vertexBuffer, bufferSize);
  vkDestroyBuffer(device, stagingBuffer, NULL);
  vkFreeMemory(device, stagingBufferMemory, NULL);
}

void create_index_buffer() {
  VkDeviceSize bufferSize = sizeof(indices[0]) * arrlen(indices);

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &stagingBuffer, &stagingBufferMemory);

  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, indices, (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  create_buffer(
      bufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &indexBuffer, &indexBufferMemory);

  copy_buffer(stagingBuffer, indexBuffer, bufferSize);

  vkDestroyBuffer(device, stagingBuffer, NULL);
  vkFreeMemory(device, stagingBufferMemory, NULL);
}

void create_uniform_buffers() {
  VkDeviceSize bufferSize = sizeof(UniformBufferObject);
  arrsetlen(uniformBuffers, MAX_FRAMES_IN_FLIGHT);
  arrsetlen(uniformBuffersMemory, MAX_FRAMES_IN_FLIGHT);
  arrsetlen(uniformBuffersMapped, MAX_FRAMES_IN_FLIGHT);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    create_buffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &uniformBuffers[i], &uniformBuffersMemory[i]);
    vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0,
                &uniformBuffersMapped[i]);
  }
}

void create_descriptor_pool() {

  VkDescriptorPoolSize poolSizes[2] = {0};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = (MAX_FRAMES_IN_FLIGHT);
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = (MAX_FRAMES_IN_FLIGHT);

  VkDescriptorPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = ((sizeof poolSizes / sizeof poolSizes[0]));
  poolInfo.pPoolSizes = poolSizes;
  poolInfo.maxSets = (MAX_FRAMES_IN_FLIGHT);

  if (vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool) !=
      VK_SUCCESS) {
    perror("failed to create descriptor pool!");
    exit(EXIT_FAILURE);
  }
}

void create_descriptor_sets() {
  VkDescriptorSetLayout *layouts = NULL;
  arrsetlen(layouts, MAX_FRAMES_IN_FLIGHT);
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    layouts[i] = descriptorSetLayout;
  }

  VkDescriptorSetAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = (uint32_t)(MAX_FRAMES_IN_FLIGHT);
  allocInfo.pSetLayouts = layouts;

  arrsetlen(descriptorSets, MAX_FRAMES_IN_FLIGHT);
  if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets) !=
      VK_SUCCESS) {
    perror("failed to allocate descriptor sets!");
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

    VkDescriptorBufferInfo bufferInfo = {0};
    bufferInfo.buffer = uniformBuffers[i];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    VkDescriptorImageInfo imageInfo = {0};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = textureImageView;
    imageInfo.sampler = textureSampler;

    VkWriteDescriptorSet descriptorWrites[2] = {0};

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSets[i];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSets[i];
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(
        device, (sizeof descriptorWrites / sizeof descriptorWrites[0]),
        descriptorWrites, 0, NULL);
  }
}

void create_descriptor_set_layout() {
  VkDescriptorSetLayoutBinding uboLayoutBinding = {0};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.descriptorCount = 1;
  uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutBinding samplerLayoutBinding = {0};
  samplerLayoutBinding.binding = 1;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.pImmutableSamplers = NULL;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutBinding bindings[] = {uboLayoutBinding,
                                             samplerLayoutBinding};
  VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = (sizeof bindings / sizeof bindings[0]);
  layoutInfo.pBindings = bindings;

  if (vkCreateDescriptorSetLayout(device, &layoutInfo, NULL,
                                  &descriptorSetLayout) != VK_SUCCESS) {
    perror("failed to create descriptor set layout!");
    exit(EXIT_FAILURE);
  }
}

void copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width,
                          uint32_t height) {
  VkCommandBuffer commandBuffer = begin_single_time_commands();

  VkBufferImageCopy region = {0};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;

  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;

  region.imageOffset = (VkOffset3D){0, 0, 0};
  region.imageExtent = (VkExtent3D){width, height, 1};

  vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  end_single_time_commands(commandBuffer);
}

void create_texture_image_view() {
  textureImageView = create_image_view(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                                       VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
}

void create_texture_sampler() {
  VkSamplerCreateInfo samplerInfo = {0};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

  samplerInfo.anisotropyEnable = VK_TRUE;

  VkPhysicalDeviceProperties properties = {0};
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);
  samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.anisotropyEnable = VK_TRUE;

  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = mipLevels;

  if (vkCreateSampler(device, &samplerInfo, NULL, &textureSampler) !=
      VK_SUCCESS) {
    perror("failed to create texture sampler!");
    exit(EXIT_FAILURE);
  }
}

void load_model() {
  const struct aiScene *scene = aiImportFile(
      "assets/spot.obj", aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                             aiProcess_JoinIdenticalVertices |
                             aiProcess_SortByPType | aiProcess_EmbedTextures);

  // If the import failed, report it
  if (NULL == scene) {
    perror(aiGetErrorString());
    exit(EXIT_FAILURE);
  }

#if 0

  {
    // First vertex
    vec3 pos = {-0.5f, -0.5f, 0.0f};
    vec3 color = {1.0f, 0.0f, 0.0f};
    vec2 texcoord = {0.0f, 0.0f};
    Vertex vert1 = {0};
    glm_vec3_copy(vert1.pos, pos);
    glm_vec3_copy(vert1.color, color);
    glm_vec2_copy(vert1.texCoord, texcoord);
    arrput(vertices, vert1);
  }

  { // Second vertex
    vec3 pos = {0.5f, -0.5f, 0.0f};
    vec3 color = {0.0f, 1.0f, 0.0f};
    vec2 texcoord = {1.0f, 0.0f};
    Vertex vert2 = {0};
    glm_vec3_copy(vert2.pos, pos);
    glm_vec3_copy(vert2.color, color);
    glm_vec2_copy(vert2.texCoord, texcoord);
    arrput(vertices, vert2);
  }

  { // Third vertex
    vec3 pos = {0.5f, 0.5f, 0.0f};
    vec3 color = {0.0f, 0.0f, 1.0f};
    vec2 texcoord = {1.0f, 1.0f};
    Vertex vert3 = {0};
    glm_vec3_copy(vert3.pos, pos);
    glm_vec3_copy(vert3.color, color);
    glm_vec2_copy(vert3.texCoord, texcoord);
    arrput(vertices, vert3);
  }

  { // Fourth vertex
    vec3 pos = {-0.5f, 0.5f, 0.0f};
    vec3 color = {1.0f, 1.0f, 1.0f};
    vec2 texcoord = {0.0f, 1.0f};
    Vertex vert4 = {0};
    glm_vec3_copy(vert4.pos, pos);
    glm_vec3_copy(vert4.color, color);
    glm_vec2_copy(vert4.texCoord, texcoord);
    arrput(vertices, vert4);
  }

  arrput(indices, 0);
  arrput(indices, 1);
  arrput(indices, 2);
  arrput(indices, 2);
  arrput(indices, 3);
  arrput(indices, 0);
#endif

  for (int i = 0; i < scene->mNumMeshes; i++) {
    struct aiMesh *m = scene->mMeshes[i];

    for (int idx = 0; idx < m->mNumFaces; idx++) {
      struct aiFace face = m->mFaces[idx];
      for (int y = 0; y < face.mNumIndices; y++) {
        arrput(indices, face.mIndices[y]);
      }
    }

    for (int v = 0; v < m->mNumVertices; v++) {
      vec3 pos = {m->mVertices[v].x, m->mVertices[v].y, m->mVertices[v].z};

      vec3 color = {
          1, 1,
          1}; //{m->mColors[0][v].r, m->mColors[0][v].g, m->mColors[0][v].b};
      vec2 texcoord = {m->mTextureCoords[i][v].x,
                       1.0f - m->mTextureCoords[i][v].y};

      Vertex vert = {0};
      glm_vec3_copy(pos, vert.pos);
      glm_vec3_copy(color, vert.color);
      glm_vec2_copy(texcoord, vert.texCoord);

      arrput(vertices, vert);
    }

    break;
  }
  printf("loaded %td indices\n", arrlen(indices));
  printf("loaded %td verts\n", arrlen(vertices));

  aiReleaseImport(scene);
}

void generate_mipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth,
                      int32_t texHeight, uint32_t mipLevels) {

  // Check if image format supports linear blitting
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat,
                                      &formatProperties);

  if (!(formatProperties.optimalTilingFeatures &
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
    perror("texture image format does not support linear blitting!");
    exit(EXIT_FAILURE);
  }

  VkCommandBuffer commandBuffer = begin_single_time_commands();

  VkImageMemoryBarrier barrier = {0};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.image = image;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.subresourceRange.levelCount = 1;

  int32_t mipWidth = texWidth;
  int32_t mipHeight = texHeight;

  for (uint32_t i = 1; i < mipLevels; i++) {
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                         &barrier);

    VkImageBlit blit = {0};
    blit.srcOffsets[0] = (VkOffset3D){0, 0, 0};
    blit.srcOffsets[1] = (VkOffset3D){mipWidth, mipHeight, 1};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = i - 1;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.dstOffsets[0] = (VkOffset3D){0, 0, 0};
    blit.dstOffsets[1] = (VkOffset3D){mipWidth > 1 ? mipWidth / 2 : 1,
                                      mipHeight > 1 ? mipHeight / 2 : 1, 1};
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;

    vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_LINEAR);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                         NULL, 1, &barrier);

    if (mipWidth > 1)
      mipWidth /= 2;
    if (mipHeight > 1)
      mipHeight /= 2;
  }

  barrier.subresourceRange.baseMipLevel = mipLevels - 1;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                       NULL, 1, &barrier);

  end_single_time_commands(commandBuffer);
}

void create_texture_image() {
  int texWidth, texHeight, texChannels;
  stbi_uc *pixels = stbi_load("assets/spot.png", &texWidth, &texHeight,
                              &texChannels, STBI_rgb_alpha);
  VkDeviceSize imageSize = texWidth * texHeight * 4;
  mipLevels = (uint32_t)(floor(log2(max(texWidth, texHeight)))) + 1;
  printf("total mip levels: %d\n", mipLevels);

  if (!pixels) {
    perror("failed to load texture image!");
    exit(EXIT_FAILURE);
  }

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  create_buffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &stagingBuffer, &stagingBufferMemory);
  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
  memcpy(data, pixels, imageSize);
  vkUnmapMemory(device, stagingBufferMemory);
  stbi_image_free(pixels);

  create_image(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT,
               VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &textureImage,
               &textureImageMemory);

  transition_image_layout(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
  copy_buffer_to_image(stagingBuffer, textureImage, texWidth, texHeight);

  /*  transition_image_layout(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  mipLevels);*/
  generate_mipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight,
                   mipLevels);
  vkDestroyBuffer(device, stagingBuffer, NULL);
  vkFreeMemory(device, stagingBufferMemory, NULL);
}

static void framebuffer_resize_callback(GLFWwindow *window, int width,
                                        int height) {
  // globals = glfwGetWindowUserPointer(window);
  framebufferResized = true;
}

int main(int argc, char *argv[]) {
  set_pwd_to_exe_dir();
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  window = glfwCreateWindow(screenWidth, screenHeight, "VKT", NULL, NULL);

  // NOTE: We can use this for the resize to get the state when we are not
  // using globals any more glfwSetWindowUserPointer(window, this);

  glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);

  check_extension_support();

  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Test";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "VKT";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  char **glfwExtensions = get_required_extensions();

  createInfo.ppEnabledExtensionNames = (const char **)glfwExtensions;
  createInfo.enabledExtensionCount = arrlen(glfwExtensions);

  if (!check_validation_layer_support()) {
    printf("validation support not avilible\n");
    createInfo.enabledLayerCount = 0;
  } else {
    char *layer = "VK_LAYER_KHRONOS_validation";
    arrput(layers, layer);
    createInfo.ppEnabledLayerNames = (const char *const *)layers;
    createInfo.enabledLayerCount = arrlen(layers);
    printf("enabled validation layer: [%d]%s\n", createInfo.enabledLayerCount,
           createInfo.ppEnabledLayerNames[0]);
  }

  VkResult err = vkCreateInstance(&createInfo, NULL, &instance);
  if (err != VK_SUCCESS) {
    printf("Error: %d\n", err);
    perror("failed to create instance");
    return EXIT_FAILURE;
  }
  arrfree(glfwExtensions);

  create_surface();

  pick_physical_device();
  create_logical_device();
  create_queues_for_device(device);
  create_swap_chain();
  create_image_views();
  create_render_pass();
  create_descriptor_set_layout();
  create_graphics_pipeline();
  create_command_pool();
  create_color_resources();
  create_depth_resources();
  create_framebuffers();
  create_texture_image();
  create_texture_image_view();
  create_texture_sampler();

  load_model();
  create_vertex_buffer();
  create_index_buffer();
  create_uniform_buffers();
  create_descriptor_pool();
  create_descriptor_sets();

  create_command_buffers();
  create_sync_objects();

  while (!glfwWindowShouldClose(window) &&
         glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) {
    glfwPollEvents();
    draw_frame();
  }

  vkDeviceWaitIdle(device);
  cleanup_swap_chain();

  vkDestroySampler(device, textureSampler, NULL);
  vkDestroyImageView(device, textureImageView, NULL);
  vkDestroyImage(device, textureImage, NULL);
  vkFreeMemory(device, textureImageMemory, NULL);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroyBuffer(device, uniformBuffers[i], NULL);
    vkFreeMemory(device, uniformBuffersMemory[i], NULL);
  }
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
  vkDestroyDescriptorPool(device, descriptorPool, NULL);
  vkDestroyBuffer(device, indexBuffer, NULL);
  vkFreeMemory(device, indexBufferMemory, NULL);
  vkDestroyBuffer(device, vertexBuffer, NULL);
  vkFreeMemory(device, vertexBufferMemory, NULL);
  vkDestroyPipeline(device, graphicsPipeline, NULL);
  vkDestroyPipelineLayout(device, pipelineLayout, NULL);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(device, renderFinishedSemaphores[i], NULL);
    vkDestroySemaphore(device, imageAvailableSemaphores[i], NULL);
    vkDestroyFence(device, inFlightFences[i], NULL);
  }

  vkDestroyCommandPool(device, commandPool, NULL);
  vkDestroyRenderPass(device, renderPass, NULL);
  vkDestroyDevice(device, NULL);
  vkDestroySurfaceKHR(instance, surface, NULL);
  vkDestroyInstance(instance, NULL);
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
